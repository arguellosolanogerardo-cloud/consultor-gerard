"""
Sistema optimizado para construcción de índice FAISS con rate limiting robusto.
Diseñado para manejar grandes volúmenes de documentos sin exceder límites de API.
"""

import os
import json
import time
import hashlib
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from tqdm import tqdm
import faiss


@dataclass
class BuilderConfig:
    """Configuración del constructor de índice FAISS"""
    rate_limit_per_minute: int = 60
    batch_size: int = 50
    save_every: int = 500
    delay_between_requests: float = 1.2
    max_retries: int = 5
    initial_backoff: float = 2.0
    max_backoff: float = 60.0
    checkpoint_file: str = "faiss_checkpoint.json"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Checkpoint:
    """Estado del proceso de construcción para recuperación"""
    processed_chunks: int
    total_chunks: int
    last_saved_at: int
    timestamp: str
    config: Dict[str, Any]
    
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> Optional['Checkpoint']:
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cls(**data)
        except Exception as e:
            print(f"⚠️ No se pudo cargar checkpoint: {e}")
            return None


class RateLimiter:
    """
    Control de rate limiting con ventana deslizante.
    Previene exceder límites de API mediante tracking temporal de requests.
    """
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.request_times = deque(maxlen=requests_per_minute)
        self.window_seconds = 60.0
    
    def wait_if_needed(self):
        """Pausa la ejecución si se excedería el límite de rate"""
        now = time.time()
        
        # Limpiar requests antiguos fuera de la ventana
        cutoff = now - self.window_seconds
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
        
        # Si alcanzamos el límite, esperar hasta que expire el más antiguo
        if len(self.request_times) >= self.requests_per_minute:
            oldest = self.request_times[0]
            wait_time = self.window_seconds - (now - oldest) + 0.1  # +0.1s margen
            if wait_time > 0:
                print(f"⏳ Rate limit alcanzado. Esperando {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        # Registrar este request
        self.request_times.append(time.time())


class FAISSVectorBuilder:
    """
    Constructor robusto de índice FAISS con capacidades de:
    - Rate limiting inteligente
    - Reintentos con backoff exponencial
    - Guardado incremental
    - Recuperación desde checkpoint
    """
    
    def __init__(self, config: BuilderConfig, embedding_function: Callable):
        self.config = config
        self.embedding_function = embedding_function
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self.index: Optional[faiss.Index] = None
        self.processed_count = 0
        
    def _exponential_backoff(self, attempt: int) -> float:
        """Calcula tiempo de espera con backoff exponencial"""
        wait = min(
            self.config.initial_backoff * (2 ** attempt),
            self.config.max_backoff
        )
        # Agregar jitter aleatorio (±20%)
        jitter = wait * 0.2 * (2 * np.random.random() - 1)
        return wait + jitter
    
    def _embed_with_retry(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings con reintentos automáticos y backoff exponencial.
        Maneja errores de API y rate limiting.
        """
        for attempt in range(self.config.max_retries):
            try:
                # Esperar si es necesario por rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Delay adicional entre requests
                if attempt > 0:
                    wait_time = self._exponential_backoff(attempt)
                    print(f"🔄 Reintento {attempt + 1}/{self.config.max_retries} después de {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(self.config.delay_between_requests)
                
                # Llamar a la función de embedding
                embeddings = self.embedding_function(texts)
                
                # Convertir a numpy array si es necesario
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings, dtype=np.float32)
                
                return embeddings
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Errores de rate limit (429)
                if '429' in error_msg or 'quota' in error_msg or 'rate' in error_msg:
                    wait_time = self._exponential_backoff(attempt + 1)
                    print(f"⚠️ Rate limit excedido. Esperando {wait_time:.1f}s antes de reintentar...")
                    time.sleep(wait_time)
                    continue
                
                # Errores de timeout o conexión
                elif 'timeout' in error_msg or 'connection' in error_msg:
                    print(f"⚠️ Error de conexión: {e}")
                    if attempt < self.config.max_retries - 1:
                        continue
                
                # Otros errores
                else:
                    print(f"❌ Error en embedding: {e}")
                    if attempt < self.config.max_retries - 1:
                        continue
                    else:
                        raise
        
        raise Exception(f"❌ Falló después de {self.config.max_retries} intentos")
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """
        Crea índice FAISS optimizado para CPU.
        Usa IndexFlatL2 para máxima precisión.
        """
        print(f"📐 Creando índice FAISS con dimensión {dimension}")
        index = faiss.IndexFlatL2(dimension)
        return index
    
    def _save_index(self, filepath: str):
        """Guarda el índice FAISS en disco"""
        if self.index is None:
            print("⚠️ No hay índice para guardar")
            return
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Guardar índice
        faiss.write_index(self.index, filepath)
        print(f"💾 Índice guardado: {filepath} ({self.index.ntotal} vectores)")
    
    def _save_checkpoint(self, total_chunks: int):
        """Guarda checkpoint del progreso actual"""
        checkpoint = Checkpoint(
            processed_chunks=self.processed_count,
            total_chunks=total_chunks,
            last_saved_at=self.index.ntotal if self.index else 0,
            timestamp=datetime.now().isoformat(),
            config=self.config.to_dict()
        )
        checkpoint.save(self.config.checkpoint_file)
        print(f"📌 Checkpoint guardado: {self.processed_count}/{total_chunks} chunks procesados")
    
    def build_from_documents(
        self,
        documents: List[Any],
        output_path: str = "faiss_index/index.faiss",
        resume_from_checkpoint: bool = True
    ) -> faiss.Index:
        """
        Construye índice FAISS desde documentos con capacidad de reanudar.
        
        Args:
            documents: Lista de documentos (deben tener .page_content)
            output_path: Ruta donde guardar el índice
            resume_from_checkpoint: Si True, intenta reanudar desde checkpoint
        
        Returns:
            Índice FAISS construido
        """
        # Extraer textos de los documentos
        texts = [doc.page_content for doc in documents]
        total_chunks = len(texts)
        
        print(f"\n{'='*60}")
        print(f"🚀 CONSTRUCCIÓN DE ÍNDICE FAISS")
        print(f"{'='*60}")
        print(f"📊 Total de chunks: {total_chunks}")
        print(f"⚙️ Configuración:")
        print(f"   - Rate limit: {self.config.rate_limit_per_minute} req/min")
        print(f"   - Batch size: {self.config.batch_size}")
        print(f"   - Delay entre requests: {self.config.delay_between_requests}s")
        print(f"   - Guardar cada: {self.config.save_every} chunks")
        print(f"   - Max reintentos: {self.config.max_retries}")
        
        # Intentar cargar checkpoint
        start_index = 0
        if resume_from_checkpoint:
            checkpoint = Checkpoint.load(self.config.checkpoint_file)
            if checkpoint:
                print(f"\n📂 Checkpoint encontrado:")
                print(f"   - Procesados: {checkpoint.processed_chunks}/{checkpoint.total_chunks}")
                print(f"   - Timestamp: {checkpoint.timestamp}")
                
                # Cargar índice parcial si existe
                if os.path.exists(output_path):
                    try:
                        self.index = faiss.read_index(output_path)
                        start_index = checkpoint.processed_chunks
                        self.processed_count = start_index
                        print(f"✅ Índice parcial cargado. Reanudando desde chunk {start_index}")
                    except Exception as e:
                        print(f"⚠️ No se pudo cargar índice parcial: {e}")
                        print("🔄 Iniciando desde cero...")
        
        # Procesar en lotes
        print(f"\n{'='*60}")
        print(f"⚡ PROCESAMIENTO DE EMBEDDINGS")
        print(f"{'='*60}\n")
        
        with tqdm(total=total_chunks, initial=start_index, desc="Procesando chunks") as pbar:
            for i in range(start_index, total_chunks, self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_num = (i // self.config.batch_size) + 1
                total_batches = (total_chunks + self.config.batch_size - 1) // self.config.batch_size
                
                try:
                    # Generar embeddings con reintentos
                    embeddings = self._embed_with_retry(batch_texts)
                    
                    # Normalizar vectores (importante para L2)
                    faiss.normalize_L2(embeddings)
                    
                    # Crear índice si es el primero
                    if self.index is None:
                        dimension = embeddings.shape[1]
                        self.index = self._create_index(dimension)
                    
                    # Agregar al índice
                    self.index.add(embeddings)
                    self.processed_count += len(batch_texts)
                    
                    # Actualizar barra de progreso
                    pbar.update(len(batch_texts))
                    pbar.set_postfix({
                        'batch': f'{batch_num}/{total_batches}',
                        'vectores': self.index.ntotal
                    })
                    
                    # Guardar incrementalmente
                    if self.processed_count % self.config.save_every == 0:
                        self._save_index(output_path)
                        self._save_checkpoint(total_chunks)
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️ Interrupción detectada. Guardando progreso...")
                    self._save_index(output_path)
                    self._save_checkpoint(total_chunks)
                    print("✅ Progreso guardado. Puedes reanudar más tarde.")
                    raise
                
                except Exception as e:
                    print(f"\n❌ Error procesando batch {batch_num}: {e}")
                    # Guardar progreso antes de fallar
                    if self.index and self.index.ntotal > 0:
                        self._save_index(output_path)
                        self._save_checkpoint(total_chunks)
                    raise
        
        # Guardado final
        print(f"\n{'='*60}")
        print(f"💾 GUARDADO FINAL")
        print(f"{'='*60}")
        self._save_index(output_path)
        
        # Limpiar checkpoint
        if os.path.exists(self.config.checkpoint_file):
            os.remove(self.config.checkpoint_file)
            print(f"🧹 Checkpoint eliminado (proceso completado)")
        
        print(f"\n{'='*60}")
        print(f"✅ ÍNDICE FAISS COMPLETADO")
        print(f"{'='*60}")
        print(f"📊 Total de vectores: {self.index.ntotal}")
        print(f"📁 Guardado en: {output_path}")
        print(f"⏱️ Tiempo total estimado: {(total_chunks * self.config.delay_between_requests / 60):.1f} minutos")
        
        return self.index

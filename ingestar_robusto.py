"""
Script optimizado para crear índice FAISS con rate limiting robusto.

VERSIÓN MEJORADA que maneja:
- Rate limits de Google Embeddings API (sin cortes)
- Procesamiento en lotes con reintentos automáticos
- Guardado incremental y checkpoints
- Capacidad de reanudar procesos interrumpidos

Uso:
    python ingestar_robusto.py              # Crear nuevo índice
    python ingestar_robusto.py --resume     # Reanudar proceso interrumpido
    python ingestar_robusto.py --force      # Forzar recreación completa
"""

import os
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from faiss_builder import FAISSVectorBuilder, BuilderConfig

# Cargar variables de entorno
load_dotenv()

# Verificar API Key
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR: No se encontró GOOGLE_API_KEY en variables de entorno")
    print("   Configura la variable antes de ejecutar este script")
    exit(1)

# Configuración de rutas
DATA_PATH = "documentos_srt/"
FAISS_INDEX_PATH = "faiss_index"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")
FAISS_PKL_FILE = os.path.join(FAISS_INDEX_PATH, "index.pkl")


def get_srt_documents(data_path):
    """
    Carga todos los archivos .srt del directorio especificado.
    
    Returns:
        list: Lista de documentos LangChain
    """
    print(f"\n📂 Cargando archivos .srt desde: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: El directorio {data_path} no existe")
        return []
    
    documents = []
    file_count = 0
    
    # Cargar todos los archivos .srt
    for filename in os.listdir(data_path):
        if filename.endswith('.srt'):
            filepath = os.path.join(data_path, filename)
            try:
                loader = TextLoader(filepath, encoding='utf-8')
                docs = loader.load()
                
                # Agregar metadata con el nombre del archivo
                for doc in docs:
                    doc.metadata['source'] = filename
                
                documents.extend(docs)
                file_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Error al cargar {filename}: {e}")
    
    print(f"✅ Cargados {file_count} archivos .srt ({len(documents)} documentos)")
    return documents


def split_documents_into_chunks(documents):
    """
    Divide los documentos en chunks más pequeños.
    
    Returns:
        list: Lista de chunks de documentos
    """
    print(f"\n✂️ Dividiendo documentos en chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Creados {len(chunks)} chunks (size=10000, overlap=1000)")
    
    return chunks


def create_faiss_index(text_chunks, force_recreate=False, resume=False):
    """
    Crea índice FAISS usando el builder robusto con rate limiting.
    
    Args:
        text_chunks: Lista de chunks de documentos
        force_recreate: Si True, elimina índice existente
        resume: Si True, intenta reanudar desde checkpoint
    """
    # Verificar si ya existe
    if os.path.exists(FAISS_INDEX_FILE) and not force_recreate and not resume:
        print(f"\n✅ Índice FAISS ya existe en {FAISS_INDEX_PATH}")
        print(f"   Usa --force para recrear o --resume para continuar proceso interrumpido")
        return
    
    # Eliminar índice existente si se fuerza recreación
    if force_recreate and os.path.exists(FAISS_INDEX_PATH):
        print(f"\n🗑️ Eliminando índice existente: {FAISS_INDEX_PATH}")
        shutil.rmtree(FAISS_INDEX_PATH)
    
    # Crear directorio
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    
    # Header
    print(f"\n{'='*70}")
    print(f"🚀 CONSTRUCCIÓN DE ÍNDICE FAISS CON RATE LIMITING ROBUSTO")
    print(f"{'='*70}")
    print(f"📄 Total de chunks a procesar: {len(text_chunks)}")
    print(f"🔧 Modelo embeddings: models/embedding-001")
    print(f"📐 Dimensión vectorial: 768")
    
    # Configuración robusta
    config = BuilderConfig(
        rate_limit_per_minute=50,        # 50 req/min (conservador)
        batch_size=50,                   # Lotes de 50 docs
        save_every=500,                  # Guardar cada 500 vectores
        delay_between_requests=1.5,       # 1.5s entre lotes
        max_retries=5,                   # 5 reintentos
        initial_backoff=2,               # Backoff inicial 2s
        max_backoff=60,                  # Backoff máximo 60s
        checkpoint_file='faiss_checkpoint.json'
    )
    
    print(f"\n⚙️ CONFIGURACIÓN:")
    print(f"   • Rate limit: {config.rate_limit_per_minute} peticiones/minuto")
    print(f"   • Batch size: {config.batch_size} documentos/lote")
    print(f"   • Guardar cada: {config.save_every} vectores")
    print(f"   • Delay entre lotes: {config.delay_between_requests}s")
    print(f"   • Reintentos máximos: {config.max_retries}")
    print(f"   • Backoff exponencial: {config.initial_backoff}s → {config.max_backoff}s")
    
    # Crear embeddings
    print(f"\n🔧 Inicializando GoogleGenerativeAIEmbeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Crear función de embedding para el builder
    def embed_function(texts: list) -> list:
        """Wrapper para embeddings de Google"""
        return embeddings.embed_documents(texts)
    
    # Crear builder con función de embedding
    builder = FAISSVectorBuilder(config, embed_function)
    
    try:
        print(f"\n🚀 Iniciando construcción del índice...")
        if resume:
            print(f"♻️ Modo RESUME: intentando reanudar desde checkpoint")
        
        # Construir índice (con progress bar, checkpoints, reintentos)
        index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        builder.build_from_documents(
            documents=text_chunks,
            output_path=index_file,
            resume_from_checkpoint=resume
        )
        
        # Guardar también el archivo .pkl con los documentos
        import pickle
        pkl_file = os.path.join(FAISS_INDEX_PATH, "index.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(text_chunks, f)
        print(f"💾 Documentos guardados: {pkl_file}")
        
        # Éxito - mostrar resumen
        print(f"\n{'='*70}")
        print(f"✅ ÍNDICE FAISS CREADO EXITOSAMENTE")
        print(f"{'='*70}")
        print(f"📁 Ubicación: {FAISS_INDEX_PATH}")
        print(f"📄 Archivos:")
        print(f"   • {FAISS_INDEX_FILE}")
        print(f"   • {FAISS_PKL_FILE}")
        
        # Mostrar tamaños
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_PKL_FILE):
            size_faiss = os.path.getsize(FAISS_INDEX_FILE) / (1024*1024)
            size_pkl = os.path.getsize(FAISS_PKL_FILE) / (1024*1024)
            print(f"\n📊 Tamaños:")
            print(f"   • index.faiss: {size_faiss:.2f} MB")
            print(f"   • index.pkl: {size_pkl:.2f} MB")
            print(f"   • Total: {size_faiss + size_pkl:.2f} MB")
        
        print(f"\n💡 SIGUIENTE PASO:")
        print(f"   Ejecuta consultar_web.py o consultar_terminal.py para usar el índice")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ PROCESO INTERRUMPIDO POR USUARIO (Ctrl+C)")
        print(f"💾 Progreso guardado en checkpoint")
        print(f"🔄 Para continuar donde quedó, ejecuta:")
        print(f"   python ingestar_robusto.py --resume")
        
    except Exception as e:
        print(f"\n\n❌ ERROR DURANTE LA CONSTRUCCIÓN:")
        print(f"   {str(e)}")
        print(f"💾 Progreso guardado en checkpoint")
        print(f"🔄 Para reintentar desde donde falló, ejecuta:")
        print(f"   python ingestar_robusto.py --resume")
        raise


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Crea índice FAISS robusto con rate limiting para archivos .srt"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la recreación del índice (elimina el existente)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda proceso interrumpido desde último checkpoint"
    )
    
    args = parser.parse_args()
    
    # Banner inicial
    print(f"\n{'='*70}")
    print(f"📚 INGESTIÓN DE DOCUMENTOS .SRT → ÍNDICE FAISS")
    print(f"{'='*70}")
    
    # Paso 1: Cargar documentos
    documents = get_srt_documents(DATA_PATH)
    if not documents:
        print(f"\n❌ No se encontraron documentos para procesar")
        return
    
    # Paso 2: Dividir en chunks
    text_chunks = split_documents_into_chunks(documents)
    if not text_chunks:
        print(f"\n❌ No se generaron chunks para procesar")
        return
    
    # Paso 3: Crear índice FAISS con builder robusto
    create_faiss_index(
        text_chunks=text_chunks,
        force_recreate=args.force,
        resume=args.resume
    )
    
    print(f"\n{'='*70}")
    print(f"✅ PROCESO COMPLETADO")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

"""
Este script se encarga de procesar documentos de texto (en formato .srt),
dividirlos en fragmentos (chunks) y crear una base de datos vectorial utilizando FAISS.
Los embeddings se generan utilizando la API de Google Generative AI.

VERSIÓN OPTIMIZADA con:
- Rate limiting robusto para evitar cortes de API
- Procesamiento en lotes con reintentos automáticos
- Guardado incremental y checkpoints
- Capacidad de reanudar si se interrumpe

Uso:
- Asegúrate de tener un archivo .env con tu GOOGLE_API_KEY.
- Ejecuta el script con `python ingestar.py`.
- Para forzar la recreación, usa `python ingestar.py --force`.
- Para reanudar proceso interrumpido, usa `python ingestar.py --resume`.
"""

import os
import shutil
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from tqdm import tqdm
from faiss_builder import FAISSVectorBuilder, BuilderConfig
import pickle

# Cargar variables de entorno
load_dotenv()

# Verificar que la API Key existe
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ No se encontró la clave de API para Google.")
    print("Por favor, configura la variable de entorno GOOGLE_API_KEY.")
    exit()

# Rutas
DATA_PATH = "documentos_srt/"
FAISS_INDEX_PATH = "faiss_index"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")
FAISS_PKL_FILE = os.path.join(FAISS_INDEX_PATH, "index.pkl")

def get_srt_text(data_path):
    """
    Carga el texto de todos los documentos .srt desde el directorio especificado.
    """
    doc_counter = 0
    documents_list = []
    print("Leyendo archivos .srt...")
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith(".srt"):
            file_path = os.path.join(data_path, filename)
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                documents_list.extend(documents)
                doc_counter += 1
            except Exception as e:
                print(f"Error al leer el archivo {filename}: {e}")
    print(f"Se cargaron {doc_counter} documentos.")
    return documents_list

def get_text_chunks(docs):
    """
    Divide los documentos en fragmentos más pequeños.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(docs)
    print(f"Documentos divididos en {len(chunks)} trozos.")
    return chunks

def create_vector_store(text_chunks):
    """
    Crea y guarda la base de datos vectorial FAISS procesando los chunks en lotes.
    PROTECCIÓN ANTI-RATE-LIMIT: Batches pequeños, pausas estratégicas, retry automático.
    """
    import time
    try:
        # Inicializar embeddings con retry
        embeddings = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    task_type="retrieval_document"
                )
                print("✅ Embeddings de Google listos")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"⚠️ Intento {attempt + 1}/{max_retries} falló: {e}")
                    print(f"   Esperando {wait_time}s antes de reintentar...")
                    time.sleep(wait_time)
                else:
                    raise
        
        print(f"Creando índice FAISS a partir de {len(text_chunks)} chunks.")
        
        # CONFIGURACIÓN ANTI-RATE-LIMIT
        batch_size = 50  # Reducido de 100 a 50
        pause_every = 5  # Pausar cada 5 batches
        pause_seconds = 3  # Pausa de 3 segundos
        total_batches = (len(text_chunks) + batch_size - 1) // batch_size
        
        vs = None
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            try:
                print(f"Procesando lote {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)
                
                if vs is None:
                    # Primer lote: crear el índice
                    vs = FAISS.from_documents(batch, embeddings)
                else:
                    # Lotes siguientes: agregar al índice existente
                    vs_batch = FAISS.from_documents(batch, embeddings)
                    vs.merge_from(vs_batch)
                
                print("✅")
                
                # PAUSA ESTRATÉGICA cada N batches
                if batch_num % pause_every == 0 and batch_num < total_batches:
                    print(f"💤 Pausa de {pause_seconds}s (evitar rate limit)...", flush=True)
                    time.sleep(pause_seconds)
            
            except Exception as batch_error:
                print(f"⚠️ Error en batch {batch_num}")
                print(f"Esperando 10 segundos y reintentando...")
                time.sleep(10)
                
                # Reintentar el batch
                try:
                    if vs is None:
                        vs = FAISS.from_documents(batch, embeddings)
                    else:
                        vs_batch = FAISS.from_documents(batch, embeddings)
                        vs.merge_from(vs_batch)
                    print(f"✅ Batch {batch_num} completado en reintento")
                except Exception as retry_error:
                    print(f"❌ ERROR FATAL en batch {batch_num}: {retry_error}")
                    print(f"Guardando progreso parcial...")
                    if vs:
                        vs.save_local(FAISS_INDEX_PATH + "_parcial")
                        print(f"⚠️ Índice parcial guardado: {FAISS_INDEX_PATH}_parcial")
                    raise
        
        # Guardar el índice completo
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        vs.save_local(FAISS_INDEX_PATH)
        print(f"¡Éxito! Índice FAISS creado con {len(text_chunks)} chunks y guardado en '{FAISS_INDEX_PATH}'.")
    except Exception as e:
        print(f"Ocurrió un error durante la creación del índice FAISS: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Función principal que orquesta la carga y procesamiento de documentos.
    """
    parser = argparse.ArgumentParser(description="Procesa documentos .srt y crea una base de datos vectorial ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Fuerza la eliminación de la base de datos existente sin pedir confirmación.")
    args = parser.parse_args()

    if os.path.exists(FAISS_INDEX_PATH):
        if args.force:
            print("Opción --force detectada. Borrando índice FAISS existente...")
            shutil.rmtree(FAISS_INDEX_PATH)
            print("Índice borrado.")
        else:
            respuesta = input(f"El índice en '{FAISS_INDEX_PATH}' ya existe. ¿Deseas borrarlo y volver a creararlo? (s/n): ").lower()
            if respuesta == 's':
                print("Borrando índice existente...")
                shutil.rmtree(FAISS_INDEX_PATH)
                print("Índice borrado.")
            else:
                print("Proceso cancelado. No se han realizado cambios.")
                return

    print("Iniciando el proceso de ingesta de documentos...")
    raw_docs = get_srt_text(DATA_PATH)
    
    if raw_docs:
        print("Dividiendo documentos en trozos...")
        text_chunks = get_text_chunks(raw_docs)
        create_vector_store(text_chunks)

if __name__ == "__main__":
    main()


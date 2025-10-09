"""
Este script se encarga de procesar documentos de texto (en formato .srt),
dividirlos en fragmentos (chunks) y crear una base de datos vectorial utilizando ChromaDB.
Los embeddings se generan utilizando la API de Google Generative AI.

Funcionalidades:
- Carga de variables de entorno para la clave de API de Google.
- Lee todos los archivos .srt de un directorio especificado.
- Divide el texto de los documentos en fragmentos más pequeños para su procesamiento.
- Genera embeddings para cada fragmento de texto y los almacena en una base de datos ChromaDB.
- Guarda la base de datos en el disco para su uso posterior.
- Maneja los límites de tasa de la API introduciendo pausas entre solicitudes.

Uso:
- Asegúrate de tener un archivo .env con tu GOOGLE_API_KEY.
- Ejecuta el script con `python ingestar.py`.
- Para forzar la recreación de la base de datos, usa `python ingestar.py --force`.
"""

import os
import shutil
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from tqdm import tqdm

# Cargar variables de entorno
load_dotenv()

# Verificar que la API Key existe
if not os.getenv("GOOGLE_API_KEY"):
    print("No se encontró la clave de API para Google. Por favor, configura la variable de entorno GOOGLE_API_KEY.")
    exit()

# Rutas
DATA_PATH = "documentos_srt/"
FAISS_INDEX_PATH = "faiss_index"

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
    Crea y guarda la base de datos vectorial FAISS procesando los chunks.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print(f"Creando índice FAISS a partir de {len(text_chunks)} chunks. Esto puede tardar...")
        vs = FAISS.from_documents(text_chunks, embeddings)
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        vs.save_local(FAISS_INDEX_PATH)
        print(f"¡Éxito! Índice FAISS creado y guardado en '{FAISS_INDEX_PATH}'.")
    except Exception as e:
        print(f"Ocurrió un error durante la creación del índice FAISS: {e}")

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


"""
Este script se encarga de procesar documentos de texto (en formato .srt),
dividirlos en fragmentos (chunks) y crear una base de datos vectorial utilizando FAISS.
Los embeddings se generan utilizando la API de Google Generative AI.

Funcionalidades:
- Carga de variables de entorno para la clave de API de Google.
- Lee todos los archivos .srt de un directorio especificado.
- Divide el texto de los documentos en fragmentos más pequeños para su procesamiento.
- Genera embeddings para cada fragmento de texto y los almacena en un índice FAISS.
- Guarda el índice FAISS en el disco para su uso posterior.
- Maneja los límites de tasa de la API introduciendo pausas entre solicitudes.

Uso:
- Asegúrate de tener un archivo .env con tu GOOGLE_API_KEY.
- Ejecuta el script con `python ingestar.py`.
- El script creará un directorio 'faiss_index' con la base de datos vectorial.
"""

import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from tqdm import tqdm

# Cargar variables de entorno. Langchain usará GOOGLE_API_KEY automáticamente.
load_dotenv()

# Verificar que la API Key existe
if not os.getenv("GOOGLE_API_KEY"):
    print("No se encontró la clave de API para Google. Por favor, configura la variable de entorno GOOGLE_API_KEY.")
    exit()

# Ruta al directorio con los documentos .srt
DATA_PATH = "documentos_srt/"
FAISS_INDEX_PATH = "faiss_index"

def get_srt_text(data_path):
    """
    Carga el texto de todos los documentos .srt desde el directorio especificado.
    """
    doc_counter = 0
    documents_list = []
    print("Leyendo archivos .srt...")
    # Usar tqdm para mostrar el progreso de la carga de archivos
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

def get_vector_store(text_chunks):
    """
    Crea y guarda la base de datos vectorial FAISS procesando un chunk a la vez.
    Este método es lento pero seguro para evitar exceder los límites de la API.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        print("Creando base de datos vectorial (modo lento y seguro).")
        print("Este proceso puede tardar varias horas. Por favor, ten paciencia.")
        
        # Inicializar FAISS con el primer chunk para tener una base
        print("Procesando el primer trozo...")
        vector_store = FAISS.from_documents([text_chunks[0]], embeddings)
        
        # Pausa para asegurar que no empezamos con ráfagas de solicitudes
        time.sleep(1.2)

        # Iterar sobre el resto de los chunks, uno por uno, con una barra de progreso
        print("Procesando el resto de los trozos...")
        for chunk in tqdm(text_chunks[1:]):
            vector_store.add_documents([chunk])
            # Pausa de 1.2 segundos para estar por debajo del límite de 60 solicitudes/minuto
            time.sleep(1.2) 

        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"¡Éxito! Base de datos vectorial creada y guardada en '{FAISS_INDEX_PATH}'.")

    except Exception as e:
        print(f"Ocurrió un error durante la creación de la base de datos: {e}")

def main():
    """
    Función principal que orquesta la carga y procesamiento de documentos.
    """
    # Si el índice ya existe, no hacemos nada.
    if os.path.exists(FAISS_INDEX_PATH):
        print("La base de datos vectorial ya existe. No se necesita hacer nada.")
        return

    print("Iniciando el proceso de ingesta de documentos...")
    raw_docs = get_srt_text(DATA_PATH)
    
    if raw_docs:
        print("Dividiendo documentos en trozos...")
        text_chunks = get_text_chunks(raw_docs)
        get_vector_store(text_chunks)

if __name__ == "__main__":
    main()


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
- Maneja los límites de tasa de la API introduciendo pausas entre las solicitudes.

Uso:
- Asegúrate de tener un archivo .env con tu GOOGLE_API_KEY.
- Ejecuta el script con `python ingestar.py`.
- El script creará un directorio 'faiss_index' con la base de datos vectorial.
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from dotenv import load_dotenv
import google.generativeai as genai
import time
from tqdm import tqdm

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configurar la API de Google
if api_key:
    genai.configure(api_key=api_key)
else:
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
    full_text = ""
    for filename in os.listdir(data_path):
        if filename.endswith(".srt"):
            file_path = os.path.join(data_path, filename)
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                full_text += "\n" + documents[0].page_content
                doc_counter += 1
            except Exception as e:
                print(f"Error al leer el archivo {filename}: {e}")
    print(f"Se cargaron {doc_counter} documentos.")
    return full_text

def get_text_chunks(text):
    """
    Divide el texto en fragmentos más pequeños.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print(f"Documentos divididos en {len(chunks)} trozos.")
    return chunks

def get_vector_store(text_chunks):
    """
    Crea y guarda la base de datos vectorial FAISS a partir de los fragmentos de texto.
    Maneja los límites de tasa de la API con pausas y muestra el progreso.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Procesar en lotes para evitar sobrecargar la API
        batch_size = 50
        vector_store = None

        print("Creando base de datos vectorial. Esto puede tardar varios minutos...")
        
        # Usar tqdm para mostrar una barra de progreso
        for i in tqdm(range(0, len(text_chunks), batch_size)):
            batch = text_chunks[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                vector_store.add_texts(batch)
            
            # Pausa para respetar el límite de tasa de la API (ej. 60 solicitudes por minuto)
            time.sleep(1) 

        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"Base de datos vectorial creada y guardada en '{FAISS_INDEX_PATH}'.")

    except Exception as e:
        print(f"Ocurrió un error durante la creación de la base de datos: {e}")

def main():
    """
    Función principal que orquesta la carga y procesamiento de documentos.
    """
    print("Cargando documentos .srt...")
    raw_text = get_srt_text(DATA_PATH)
    
    if raw_text:
        print("Dividiéndolos en trozos...")
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

if __name__ == "__main__":
    main()

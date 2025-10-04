import os
from dotenv import load_dotenv

# --- CAMBIOS IMPORTANTES AQUÍ ---
# Ahora importamos TextLoader para forzar la lectura como texto
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. Carga la API Key ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
    exit()

# --- 2. Carga de los Documentos (VERSIÓN CORREGIDA) ---
print("Cargando documentos en modo texto...")

# Le decimos a DirectoryLoader que use TextLoader para todos los archivos.
# Añadimos encoding='utf-8' para manejar correctamente acentos y caracteres especiales.
loader = DirectoryLoader(
    './documentos_srt/', 
    glob="**/*.srt", 
    loader_cls=TextLoader, 
    loader_kwargs={'encoding': 'utf-8'}
)

docs = loader.load()

if not docs:
    print("No se encontraron documentos .srt en la carpeta especificada.")
    exit()

print(f"Se cargaron {len(docs)} documentos.")

# --- 3. División del Texto ---
print("Dividiendo documentos en trozos...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"Documentos divididos en {len(splits)} trozos.")

# --- 4. Creación de la Base de Datos Vectorial (ChromaDB) ---
print("Creando la base de datos vectorial con ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="./chroma_db"
)

print("¡Proceso completado! La base de conocimiento ha sido creada y guardada en 'chroma_db'.")
"""
Script de prueba para validar faiss_builder con un subset pequeño de documentos.
Esto nos permite verificar que el sistema funciona antes de procesar miles de chunks.
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from faiss_builder import FAISSVectorBuilder, BuilderConfig

# Cargar variables de entorno
load_dotenv()

# Verificar API Key
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR: No se encontró GOOGLE_API_KEY")
    exit(1)

# Configuración de prueba
DATA_PATH = "documentos_srt/"
TEST_INDEX_PATH = "faiss_index_test"
MAX_FILES = 5  # Solo procesar primeros 5 archivos para prueba

print(f"\n{'='*70}")
print(f"🧪 PRUEBA DE FAISS BUILDER (MODO TEST)")
print(f"{'='*70}\n")

# Paso 1: Cargar solo primeros archivos
print(f"📂 Cargando primeros {MAX_FILES} archivos .srt...")
documents = []
file_count = 0

for filename in os.listdir(DATA_PATH):
    if filename.endswith('.srt') and file_count < MAX_FILES:
        filepath = os.path.join(DATA_PATH, filename)
        try:
            loader = TextLoader(filepath, encoding='utf-8')
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = filename
            
            documents.extend(docs)
            file_count += 1
            print(f"   ✅ {filename}")
            
        except Exception as e:
            print(f"   ⚠️ Error en {filename}: {e}")

print(f"\n✅ Cargados {file_count} archivos ({len(documents)} documentos)")

# Paso 2: Dividir en chunks
print(f"\n✂️ Dividiendo en chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=1000
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Creados {len(chunks)} chunks")

# Paso 3: Configuración conservadora para prueba
config = BuilderConfig(
    rate_limit_per_minute=30,        # Muy conservador para prueba
    batch_size=10,                   # Lotes muy pequeños
    save_every=50,                   # Guardar frecuentemente
    delay_between_requests=2.0,       # 2 segundos entre lotes
    max_retries=3,
    initial_backoff=2,
    max_backoff=30,
    checkpoint_file='faiss_checkpoint_test.json'
)

print(f"\n⚙️ Configuración de prueba:")
print(f"   • Rate limit: {config.rate_limit_per_minute} req/min")
print(f"   • Batch size: {config.batch_size}")
print(f"   • Delay: {config.delay_between_requests}s")

# Paso 4: Crear embeddings y builder
print(f"\n🔧 Inicializando embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def embed_function(texts: list) -> list:
    """Wrapper para embeddings"""
    return embeddings.embed_documents(texts)

builder = FAISSVectorBuilder(config, embed_function)

# Paso 5: Construir índice de prueba
print(f"\n🚀 Construyendo índice de prueba...")
print(f"   (Esto procesará {len(chunks)} chunks con delays entre lotes)")
print(f"   Tiempo estimado: ~{(len(chunks) * config.delay_between_requests / 60):.1f} minutos\n")

try:
    import shutil
    if os.path.exists(TEST_INDEX_PATH):
        shutil.rmtree(TEST_INDEX_PATH)
    os.makedirs(TEST_INDEX_PATH, exist_ok=True)
    
    index_file = os.path.join(TEST_INDEX_PATH, "index.faiss")
    builder.build_from_documents(
        documents=chunks,
        output_path=index_file,
        resume_from_checkpoint=False
    )
    
    # Guardar documentos
    import pickle
    pkl_file = os.path.join(TEST_INDEX_PATH, "index.pkl")
    with open(pkl_file, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"\n{'='*70}")
    print(f"✅ PRUEBA EXITOSA!")
    print(f"{'='*70}")
    print(f"📁 Índice de prueba guardado en: {TEST_INDEX_PATH}")
    print(f"\n💡 Si todo funcionó correctamente, puedes ejecutar:")
    print(f"   python ingestar_robusto.py --force")
    print(f"   (para procesar TODOS los documentos)")
    
except KeyboardInterrupt:
    print(f"\n⚠️ Prueba interrumpida por usuario")
    
except Exception as e:
    print(f"\n❌ ERROR EN PRUEBA:")
    print(f"   {str(e)}")
    import traceback
    traceback.print_exc()
    print(f"\n💡 Revisa la configuración y la API key antes de continuar")

print(f"\n{'='*70}\n")

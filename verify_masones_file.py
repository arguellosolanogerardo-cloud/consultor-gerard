"""
Verificar si el archivo 'los masones.quienes son.srt' está en el índice FAISS
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY no está configurada")

print("Cargando índice FAISS...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"✅ FAISS cargado: {vectorstore.index.ntotal} documentos\n")

# Buscar el archivo específico
target_file = "los masones.quienes son"
print(f"🔍 Buscando chunks del archivo: '{target_file}'...\n")

# Recuperar todos los documentos (esto puede ser lento)
docstore = vectorstore.docstore._dict
matching_chunks = []

for doc_id, doc in docstore.items():
    source = doc.metadata.get("source", "")
    if target_file.lower() in source.lower():
        matching_chunks.append(doc)

if not matching_chunks:
    print(f"❌ NO SE ENCONTRARON chunks del archivo '{target_file}'")
    print(f"\nBuscando archivos que contengan 'mason'...")
    
    mason_files = set()
    for doc_id, doc in docstore.items():
        source = doc.metadata.get("source", "")
        if "mason" in source.lower():
            mason_files.add(source)
    
    if mason_files:
        print(f"\n✅ Archivos encontrados con 'mason':")
        for f in sorted(mason_files):
            print(f"  - {f}")
else:
    print(f"✅ ENCONTRADOS {len(matching_chunks)} chunks del archivo '{target_file}'\n")
    
    # Buscar chunks que contengan los términos clave
    keywords = ["linaje", "bis", "trick", "jac", "hack", "cuatro razas"]
    
    print("🔎 Buscando chunks con términos clave...\n")
    for i, doc in enumerate(matching_chunks, 1):
        content = doc.page_content.lower()
        
        # Verificar si contiene alguna keyword
        found_keywords = [kw for kw in keywords if kw in content]
        
        if found_keywords:
            print(f"\n{'='*80}")
            print(f"[CHUNK {i}] ✅ Contiene: {', '.join(found_keywords)}")
            print('='*80)
            print(f"Fuente: {doc.metadata.get('source', 'N/A')}")
            print(f"Contenido ({len(doc.page_content)} chars):")
            print(doc.page_content[:500])
            print("...")

print("\n✅ Verificación completada")

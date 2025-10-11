"""
Script para diagnosticar por qué FAISS no recupera información sobre los 4 linajes
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Cargar variables de entorno
load_dotenv()

# Verificar API Key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY no está configurada")

# Cargar FAISS
print("Cargando índice FAISS...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"✅ FAISS cargado: {vectorstore.index.ntotal} documentos\n")

# Búsquedas específicas
queries = [
    "LINAJE RA, BIS, JAC TRIC Y LAS 4 RAZAS",
    "cuatro linajes cuatro razas",
    "linaje ra bis trick jac",
    "los 4 linajes de la tierra",
    "linaje bis linaje trick linaje hack",
]

for query in queries:
    print(f"\n{'='*80}")
    print(f"🔍 BÚSQUEDA: {query}")
    print('='*80)
    
    # Recuperar top-5
    docs = vectorstore.similarity_search_with_score(query, k=5)
    
    if not docs:
        print("❌ NO SE ENCONTRARON DOCUMENTOS")
        continue
    
    for i, (doc, score) in enumerate(docs, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"Fuente: {doc.metadata.get('source', 'N/A')}")
        
        # Mostrar primeros 300 caracteres
        content = doc.page_content.strip()
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"Contenido: {preview}")
        
        # Buscar términos clave en el contenido
        keywords = ["linaje", "bis", "trick", "jac", "hack", "razas", "cuatro"]
        found = [kw for kw in keywords if kw.lower() in content.lower()]
        if found:
            print(f"✅ Encontrado en contenido: {', '.join(found)}")

print("\n" + "="*80)
print("✅ Diagnóstico completado")

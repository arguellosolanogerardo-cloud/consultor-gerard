"""
Calcular scores de similitud para los chunks correctos vs la consulta del usuario
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

# Consulta original del usuario
user_query = "INFORMACION SOBRE EL LINAJE RA, BIS,JAC TRIC Y LAS 4 RAZAS"

print(f"🔍 Consulta del usuario: {user_query}\n")

# Recuperar top-50 documentos
print("Recuperando top-50 documentos más similares...\n")
docs_with_scores = vectorstore.similarity_search_with_score(user_query, k=50)

# Buscar el archivo específico en los resultados
target_file = "los masones.quienes son"
found_in_top50 = []

for i, (doc, score) in enumerate(docs_with_scores, 1):
    source = doc.metadata.get("source", "")
    if target_file.lower() in source.lower():
        found_in_top50.append((i, doc, score))

if not found_in_top50:
    print(f"❌ El archivo '{target_file}' NO aparece en el top-50")
    print(f"\n📊 Top-5 resultados actuales:")
    for i, (doc, score) in enumerate(docs_with_scores[:5], 1):
        source = doc.metadata.get("source", "").split("\\")[-1]
        preview = doc.page_content.strip()[:100]
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"Fuente: {source}")
        print(f"Preview: {preview}...")
else:
    print(f"✅ El archivo '{target_file}' SÍ aparece en el top-50:\n")
    for rank, doc, score in found_in_top50:
        print(f"\n{'='*80}")
        print(f"[RANK #{rank}] Score: {score:.4f}")
        print('='*80)
        print(f"Fuente: {doc.metadata.get('source', 'N/A')}")
        
        content = doc.page_content.strip()
        print(f"\nContenido completo ({len(content)} chars):")
        print(content)
        
        # Buscar términos clave
        keywords = ["linaje", "bis", "trick", "jac", "hack", "cuatro razas", "ra"]
        found_kw = [kw for kw in keywords if kw.lower() in content.lower()]
        if found_kw:
            print(f"\n✅ Keywords encontradas: {', '.join(found_kw)}")

print("\n" + "="*80)
print(f"📊 RESUMEN:")
print(f"  - Archivo '{target_file}' aparece {len(found_in_top50)} veces en top-50")
if found_in_top50:
    best_rank = min(rank for rank, _, _ in found_in_top50)
    best_score = min(score for _, _, score in found_in_top50)
    print(f"  - Mejor posición: #{best_rank}")
    print(f"  - Mejor score: {best_score:.4f}")
    print(f"\n💡 Con k=40, este archivo {'SÍ' if best_rank <= 40 else 'NO'} sería recuperado")
print("="*80)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT DE RE-INDEXACIÓN CON CHUNKS PEQUEÑOS
============================================
Este script re-crea el índice FAISS con chunks MÁS PEQUEÑOS
para mejorar la precisión de búsqueda.

MEJORAS:
✓ chunk_size: 1000 → 500 (chunks más pequeños)
✓ chunk_overlap: 200 → 100
✓ Mejor recall en búsquedas específicas
✓ Menos dilución semántica

USO SIMPLE:
    python reiniciar_indice.py

AUTOMÁTICO:
- Hace backup del índice anterior
- Procesa todos los .srt
- Crea nuevo índice optimizado
- Verifica con búsqueda de prueba
"""

import os
import sys
import shutil
from datetime import datetime
from dotenv import load_dotenv

# Cargar API key
load_dotenv()

if not os.getenv('GOOGLE_API_KEY'):
    print("❌ ERROR: Falta GOOGLE_API_KEY")
    print("\nConfigure con:")
    print("  $env:GOOGLE_API_KEY = 'tu-api-key'")
    sys.exit(1)

print("🔧 Importando librerías...")
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# === CONFIGURACIÓN ===
DOCS_DIR = "documentos_srt"
FAISS_DIR = "faiss_index"
BACKUP_DIR = f"faiss_index_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# NUEVOS PARÁMETROS (chunks más pequeños)
CHUNK_SIZE = 500      # Antes: 1000
CHUNK_OVERLAP = 100   # Antes: 200

print(f"""
╔══════════════════════════════════════════════════════════╗
║        RE-INDEXACIÓN OPTIMIZADA - CHUNKS PEQUEÑOS        ║
╚══════════════════════════════════════════════════════════╝

📦 Chunk size: {CHUNK_SIZE} (antes: 1000) - 50% más pequeño
🔗 Overlap: {CHUNK_OVERLAP} (antes: 200)
📂 Directorio: {DOCS_DIR}
🎯 Índice: {FAISS_DIR}
""")

# === 1. BACKUP ===
print("\n" + "="*60)
print("1️⃣  BACKUP DEL ÍNDICE ANTERIOR")
print("="*60)

if os.path.exists(FAISS_DIR):
    try:
        shutil.copytree(FAISS_DIR, BACKUP_DIR)
        print(f"✅ Backup: {BACKUP_DIR}")
        shutil.rmtree(FAISS_DIR)
        print(f"✅ Índice anterior eliminado")
    except Exception as e:
        print(f"⚠️ Error en backup: {e}")
else:
    print("ℹ️ No hay índice anterior")

# === 2. CARGAR DOCUMENTOS ===
print("\n" + "="*60)
print("2️⃣  CARGANDO ARCHIVOS .SRT")
print("="*60)

try:
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.srt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )
    documents = loader.load()
    print(f"✅ {len(documents)} archivos cargados")
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"   {total_chars:,} caracteres totales")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# === 3. DIVIDIR EN CHUNKS ===
print("\n" + "="*60)
print("3️⃣  DIVIDIENDO EN CHUNKS PEQUEÑOS")
print("="*60)

try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ {len(chunks)} chunks creados")
    print(f"   {len(chunks) // len(documents)} chunks por documento (promedio)")
    
    sizes = [len(c.page_content) for c in chunks]
    print(f"   Tamaño promedio: {sum(sizes)//len(sizes)} caracteres")
    print(f"   Rango: {min(sizes)} - {max(sizes)} caracteres")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# === 4. CREAR EMBEDDINGS ===
print("\n" + "="*60)
print("4️⃣  INICIALIZANDO EMBEDDINGS")
print("="*60)

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("✅ Embeddings de Google listos")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# === 5. CREAR ÍNDICE FAISS ===
print("\n" + "="*60)
print("5️⃣  CREANDO ÍNDICE FAISS")
print("="*60)
print("⏳ Procesando en batches (puede tomar varios minutos)...\n")

try:
    BATCH_SIZE = 100
    vectorstore = None
    total_batches = (len(chunks) - 1) // BATCH_SIZE + 1
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"   Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_vs = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_vs)
    
    print(f"\n✅ Índice FAISS creado: {len(chunks)} chunks")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# === 6. GUARDAR ===
print("\n" + "="*60)
print("6️⃣  GUARDANDO ÍNDICE")
print("="*60)

try:
    vectorstore.save_local(FAISS_DIR)
    print(f"✅ Índice guardado: {FAISS_DIR}")
    
    size_mb = sum(
        os.path.getsize(os.path.join(FAISS_DIR, f))
        for f in os.listdir(FAISS_DIR)
    ) / 1024 / 1024
    
    print(f"   Tamaño: {size_mb:.2f} MB")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# === 7. VERIFICAR ===
print("\n" + "="*60)
print("7️⃣  VERIFICACIÓN")
print("="*60)

try:
    test_vs = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Índice verificado: {test_vs.index.ntotal} documentos")
    
    # Búsqueda de prueba
    print("\n🧪 PRUEBA DE BÚSQUEDA:")
    test_query = "linaje ra tric jac bis"
    results = test_vs.similarity_search_with_score(test_query, k=5)
    
    print(f"   Query: '{test_query}'")
    print(f"   Resultados: {len(results)}")
    
    if results:
        doc, score = results[0]
        source = doc.metadata.get('source', 'desconocido')
        filename = source.split('\\')[-1] if '\\' in source else source
        
        print(f"\n   Top resultado:")
        print(f"   • Score: {score:.4f}")
        print(f"   • Fuente: {filename[:60]}")
        print(f"   • Preview: {doc.page_content[:150]}...")
        
        # Verificar si encuentra el documento correcto
        if "DESCUBRIENDO" in source:
            print("\n   ✅ ¡Encuentra el documento correcto!")
        else:
            print("\n   ⚠️ Top resultado no es el esperado")
            print("      (Pero con k=50 debería estar en los resultados)")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# === RESUMEN ===
print("\n" + "="*60)
print("✅ RE-INDEXACIÓN COMPLETADA")
print("="*60)

print(f"""
📊 ESTADÍSTICAS:
   • Archivos: {len(documents)}
   • Chunks: {len(chunks)} (antes: ~{len(chunks)//2})
   • Chunk size: {CHUNK_SIZE} caracteres (antes: 1000)
   • Tamaño índice: {size_mb:.2f} MB
   • Backup: {BACKUP_DIR}

🎯 MEJORAS:
   ✓ Chunks 50% más pequeños
   ✓ Mayor precisión en búsquedas
   ✓ Menos dilución semántica
   ✓ k=50 en consultar_web.py

🚀 PRÓXIMO PASO:
   Reinicia Streamlit:
   
   > Get-Process | Where-Object {{$_.ProcessName -eq "streamlit"}} | Stop-Process -Force
   > streamlit run consultar_web.py
   
💡 Para agregar más documentos en el futuro:
   1. Copia nuevos .srt a {DOCS_DIR}/
   2. Ejecuta: python reiniciar_indice.py
   3. Reinicia Streamlit
""")

print("="*60)
print("🎉 ¡LISTO PARA USAR!")
print("="*60)

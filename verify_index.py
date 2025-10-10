#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verificar si el archivo está en FAISS"""

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vs = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)

# Obtener todos los metadatos
all_docs = vs.docstore._dict

print(f"Total de documentos en FAISS: {len(all_docs)}\n")

# Buscar el archivo problemático
target_file = "DESCUBRIENDO LOS MENSAJES OCULTOS"

found_docs = []
for doc_id, doc in all_docs.items():
    source = doc.metadata.get('source', '')
    if target_file in source:
        found_docs.append(doc)

print(f"Documentos encontrados con '{target_file}': {len(found_docs)}\n")

if found_docs:
    print("Primeros 3 chunks de este archivo:\n")
    for i, doc in enumerate(found_docs[:3], 1):
        print(f"\n{i}. Contenido:")
        print(doc.page_content[:300])
        print("...")
else:
    print(f"❌ ¡EL ARCHIVO NO ESTÁ EN EL ÍNDICE FAISS!")
    print("\nArchivos disponibles (primeros 20):")
    for i, (doc_id, doc) in enumerate(list(all_docs.items())[:20], 1):
        source = doc.metadata.get('source', 'desconocido')
        filename = source.split('\\')[-1] if '\\' in source else source
        print(f"{i}. {filename[:80]}")

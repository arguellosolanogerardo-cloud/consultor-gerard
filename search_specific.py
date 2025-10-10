#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Buscar contenido específico en FAISS"""

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vs = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)

all_docs = vs.docstore._dict

target_file = "DESCUBRIENDO LOS MENSAJES OCULTOS"
search_terms = ["linaje ra", "Miri", "bis Trick", "descendencia"]

print(f"Buscando en chunks del archivo '{target_file}'...\n")

found = False
for doc_id, doc in all_docs.items():
    source = doc.metadata.get('source', '')
    if target_file in source:
        content = doc.page_content.lower()
        for term in search_terms:
            if term.lower() in content:
                print(f"✅ ENCONTRADO: '{term}'")
                print(f"Contenido del chunk:\n{doc.page_content}\n")
                print("="*70 + "\n")
                found = True
                break

if not found:
    print("❌ No se encontraron los términos en ningún chunk de este archivo")
    print("\nEsto significa que el archivo NO se indexó correctamente")
    print("o los chunks están fragmentados de manera que separan las palabras clave")

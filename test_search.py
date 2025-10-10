#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script para diagnosticar búsqueda vectorial"""

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

print("Inicializando embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

print("Cargando FAISS...")
vs = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)

print(f"Total documentos en FAISS: {vs.index.ntotal}\n")

# Buscar con diferentes queries
queries = [
    "linaje ra",
    "linaje ra tric jac bis",
    "descendencia bis trick jack",
    "ra y Miri"
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"BÚSQUEDA: '{query}'")
    print('='*70)
    
    results = vs.similarity_search_with_score(query, k=5)
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'desconocido')
        filename = source.split('\\')[-1] if '\\' in source else source
        
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Fuente: {filename[:80]}")
        print(f"   Contenido: {doc.page_content[:200]}...")
        
print("\n" + "="*70)
print("DIAGNÓSTICO COMPLETO")

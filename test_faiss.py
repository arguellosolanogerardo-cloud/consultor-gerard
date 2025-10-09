#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test simple para verificar el índice FAISS"""

from langchain_community.vectorstores import FAISS
import hashlib

class FakeEmbeddings:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def _text_to_vector(self, text: str):
        out_bytes = b''
        counter = 0
        while len(out_bytes) < self.dim:
            h = hashlib.sha256((text + '|' + str(counter)).encode('utf-8')).digest()
            out_bytes += h
            counter += 1
        vec = [b / 255.0 for b in out_bytes[: self.dim]]
        return vec

    def embed_documents(self, texts):
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str):
        return self._text_to_vector(text)

print("Cargando índice FAISS...")
embeddings = FakeEmbeddings()
vs = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)

print(f"✅ Índice cargado correctamente")
print(f"📊 Número de documentos: {vs.index.ntotal}")
print(f"📐 Dimensión de vectores: {vs.index.d}")

# Test de búsqueda
print("\n🔍 Probando búsqueda sobre 'María Magdalena'...")
try:
    # Usar el método embed_query directamente
    query_vector = embeddings.embed_query("María Magdalena")
    docs_and_scores = vs.similarity_search_by_vector(query_vector, k=3)
    print(f"✅ Documentos encontrados: {len(docs_and_scores)}")
    for i, doc in enumerate(docs_and_scores, 1):
        print(f"\n--- Documento {i} ---")
        print(f"Fuente: {doc.metadata.get('source', 'desconocido')}")
        print(f"Contenido (primeros 200 caracteres): {doc.page_content[:200]}...")
except Exception as e:
    print(f"❌ Error en búsqueda: {e}")

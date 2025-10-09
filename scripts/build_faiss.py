"""
Reconstruye un índice FAISS desde los archivos .srt en `documentos_srt/`.
- Hace backup de `faiss_index/` si existe (se renombra con timestamp).
- Divide los documentos en chunks y genera embeddings usando GoogleGenerativeAIEmbeddings.
- Construye y guarda un FAISS local en `faiss_index/`.

Uso:
    python scripts/build_faiss.py [--force]

Advertencia: generar embeddings puede consumir cuota/API y tardar dependiendo del número de chunks.
"""

import os
import shutil
import argparse
import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import math


DATA_PATH = "documentos_srt"
FAISS_DIR = "faiss_index"


def get_srt_documents(data_path):
    docs = []
    if not os.path.exists(data_path):
        print(f"Data path '{data_path}' does not exist.")
        return docs

    files = [f for f in os.listdir(data_path) if f.endswith('.srt')]
    print(f"Found {len(files)} .srt files.")
    for fn in tqdm(files, desc='Reading .srt files'):
        path = os.path.join(data_path, fn)
        try:
            loader = TextLoader(path, encoding='utf-8')
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return docs


def split_documents(docs):
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    print("Splitting documents into chunks...")
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def compute_embeddings_with_progress(chunks, embeddings_obj, batch_size=32):
    """Compute embeddings for chunk list in batches and show global progress (percentage).

    Returns list of vectors (one per chunk).
    """
    texts = [getattr(d, 'page_content', str(d)) for d in chunks]
    total = len(texts)
    if total == 0:
        return []

    vectors = []
    from tqdm import tqdm
    batches = math.ceil(total / batch_size)
    pbar = tqdm(total=total, desc='Embedding progress', unit='chunks', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%')

    for i in range(batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        try:
            emb_batch = embeddings_obj.embed_documents(batch_texts)
        except AttributeError:
            # Some embeddings clients use embed_documents names differently
            emb_batch = embeddings_obj.embed(batch_texts)

        vectors.extend(emb_batch)
        pbar.update(len(batch_texts))

    pbar.close()
    return vectors


def backup_existing_faiss(folder_path):
    if os.path.exists(folder_path):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dst = f"{folder_path}_backup_{ts}"
        print(f"Backing up existing '{folder_path}' to '{dst}'...")
        shutil.copytree(folder_path, dst)
        print("Backup complete.")


def build_and_save_faiss(chunks, api_key, folder_path):
    print("Building embeddings object...")
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    print("Computing embeddings for all chunks (showing global progress)...")
    vectors = compute_embeddings_with_progress(chunks, embeddings, batch_size=16)

    print("Attempting to build FAISS from precomputed embeddings...")
    vs = None
    try:
        # Try common signature: FAISS.from_embeddings(embeddings=vectors, documents=chunks, embedding=embeddings)
        vs = FAISS.from_embeddings(vectors, chunks, embeddings)
    except Exception:
        try:
            # Try alternate signature without named args
            vs = FAISS.from_embeddings(vectors, chunks)
        except Exception:
            print("FAISS.from_embeddings failed or not available; falling back to FAISS.from_documents (this will re-run embeddings).")
            vs = FAISS.from_documents(chunks, embeddings)

    print(f"Saving FAISS index to '{folder_path}'...")
    # If folder exists, remove it before saving
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    vs.save_local(folder_path)
    print("FAISS index saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Recreate index even if exists')
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print('GOOGLE_API_KEY not found in environment. Please set it in .env or environment variables.')
        return

    if os.path.exists(FAISS_DIR) and not args.force:
        print(f"FAISS directory '{FAISS_DIR}' already exists. Use --force to overwrite.")
        return

    # Get documents and chunks
    docs = get_srt_documents(DATA_PATH)
    if not docs:
        print('No documents found to index. Exiting.')
        return

    chunks = split_documents(docs)

    # Backup existing faiss dir
    if os.path.exists(FAISS_DIR):
        backup_existing_faiss(FAISS_DIR)

    # Build and save FAISS index
    build_and_save_faiss(chunks, api_key, FAISS_DIR)


if __name__ == '__main__':
    main()

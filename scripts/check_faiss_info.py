import os
import sys

faiss_index_path = r"E:/proyecto-gemini/faiss_index/index.faiss"
srt_dir = r"E:/proyecto-gemini/documentos_srt"

# Check FAISS
try:
    import faiss
    idx = faiss.read_index(faiss_index_path)
    print('FAISS_INDEX_NT', idx.ntotal)
except Exception as e:
    print('FAISS_ERROR', type(e).__name__, e)

# Count SRT files
try:
    srt_files = [f for f in os.listdir(srt_dir) if f.lower().endswith('.srt')]
    print('SRT_FILES_COUNT', len(srt_files))
    for f in srt_files[:20]:
        print('SRT_FILE', f)
except Exception as e:
    print('SRT_ERROR', type(e).__name__, e)

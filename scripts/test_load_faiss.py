import os
import traceback
from pprint import pprint
from dotenv import load_dotenv


load_dotenv()

try:
    # Mostrar si hay clave de API disponible
    api_key = os.environ.get('GOOGLE_API_KEY')
    print('GOOGLE_API_KEY present?', bool(api_key))

    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not api_key:
        raise RuntimeError(
            'No GOOGLE_API_KEY found in environment. Please ensure .env is present or export the variable before running this script.'
        )

    # Crear embeddings usando la API key
    emb = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)

    # Cargar índice FAISS desde carpeta
    folder = 'faiss_index'
    print(f'Attempting to load FAISS index from "{folder}"...')
    # Allow local deserialization because index was created locally in this repo
    vs = FAISS.load_local(folder_path=folder, embeddings=emb, allow_dangerous_deserialization=True)
    print('FAISS loaded:', type(vs))

    # Realizar búsqueda de ejemplo
    query = 'meditación bendición'  # ejemplo de búsqueda
    print('Running similarity_search for:', query)
    results = vs.similarity_search(query, k=3)
    print('Found', len(results), 'results')
    for i, r in enumerate(results, 1):
        print('\n--- RESULT', i, '---')
        # r may be a Document object with page_content and metadata
        try:
            pprint({'content': r.page_content[:400], 'metadata': r.metadata})
        except Exception:
            pprint(r)

except Exception:
    print('Error while loading or querying FAISS:')
    traceback.print_exc()
    raise

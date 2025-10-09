import os
import sys
import json
from dotenv import load_dotenv

# Asegurarse de usar el path del repo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

from consultar_terminal import build_retrieval_chain, print_json_answer, get_clean_text_from_json


def main():
    if len(sys.argv) < 2:
        print("Usage: run_query.py 'tu pregunta'")
        sys.exit(1)

    query = sys.argv[1]

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print('ERROR: GOOGLE_API_KEY no definida en el entorno.')
        sys.exit(2)

    print('Inicializando y cargando LLM/FAISS. Esto puede tardar...')
    chain = build_retrieval_chain(api_key)

    print('\nEjecutando consulta:')
    print(query)

    try:
        answer = chain.invoke({'input': query})
        print('\nRespuesta cruda (JSON esperado):')
        print(answer)

        print('\nSalida formateada:')
        print_json_answer(answer)

        # También mostrar versión limpia para el log
        clean = get_clean_text_from_json(answer)
        print('\nTexto limpio (para log):')
        print(clean)

    except Exception as e:
        print('Error al ejecutar la consulta:', e)
        sys.exit(3)


if __name__ == '__main__':
    main()

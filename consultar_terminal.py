import os
import json
import re
import colorama
import argparse
import threading
import itertools
import sys
import time
from dotenv import load_dotenv
import keyring
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# Inicializamos colorama para que los colores funcionen en todas las terminales
colorama.init(autoreset=True)

# --- Carga la API Key ---
# Nota: no inicializamos la API ni recursos de red al importar el módulo.
# Creamos una función para construir la cadena de recuperación (llm + vectorstore)
def build_retrieval_chain(api_key: str):
    """Construye y devuelve el retrieval_chain usando la API key proporcionada.

    Carga el índice FAISS persistido en `faiss_index/`.
    """
    # Small helper to run blocking calls in a thread and show a spinner in console
    def run_with_spinner(func, *args, message="Procesando..."):
        result_holder = {}

        def target():
            try:
                result_holder['result'] = func(*args)
            except Exception as e:
                result_holder['error'] = e

        thread = threading.Thread(target=target)
        thread.start()

        spinner = itertools.cycle(['|', '/', '-', '\\'])
        sys.stdout.write(message + ' ')
        sys.stdout.flush()
        try:
            while thread.is_alive():
                sys.stdout.write(next(spinner))
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write('\b')
        except KeyboardInterrupt:
            pass
        thread.join()

        sys.stdout.write('\r' + ' ' * (len(message) + 2) + '\r')

        if 'error' in result_holder:
            raise result_holder['error']
        return result_holder.get('result')

    # Load LLM and embeddings with spinner to give feedback for slow init
    llm = run_with_spinner(lambda: GoogleGenerativeAI(model="models/gemini-2.5-pro", google_api_key=api_key), message="Inicializando LLM...")
    embeddings = run_with_spinner(lambda: GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key), message="Inicializando embeddings...")

    try:
        vectorstore = run_with_spinner(lambda: FAISS.load_local(folder_path="faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True), message="Cargando índice FAISS (puede tardar)...")
    except Exception as e:
        print(f"Error cargando FAISS index: {e}")
        raise

    retriever = vectorstore.as_retriever()
    retrieval_chain = (
        {
            "context": (lambda x: x["input"]) | retriever | format_docs_with_metadata,
            "input": (lambda x: x["input"]) 
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


def get_api_key():
    """Intentar obtener la API key de varias fuentes en orden:
    1. keyring del sistema (servicio 'consultor-gerard', nombre 'google_api_key')
    2. variable de entorno GOOGLE_API_KEY
    3. archivo .env (load_dotenv ya se usa en main)
    Devuelve la cadena o None si no se encuentra.
    """
    # 1) keyring
    try:
        kr = keyring.get_password('consultor-gerard', 'google_api_key')
        if kr:
            return kr
    except Exception:
        # Si keyring falla por cualquier motivo, lo ignoramos y continuamos
        pass

    # 2) environment
    api = os.environ.get('GOOGLE_API_KEY')
    if api:
        return api

    # 3) .env ya cargado por caller
    return None

# --- PERSONALIDAD DE "GERARD" (CON PROMPT CORREGIDO) ---
prompt = ChatPromptTemplate.from_template("""
--- INICIO DE INSTRUCCIONES DE PERSONALIDAD ---
1. ROL Y PERSONA: Eres "GERARD", un analista de IA que encuentra patrones en textos.

2. CONTEXTO: Analizas archivos .srt sobre temas espirituales y narrativas ocultas.

--- REGLA DE FORMATO DE SALIDA (LA MÁS IMPORTANTE) ---
Tu única forma de responder es generando un objeto JSON. Tu respuesta DEBE ser un array de objetos JSON válido. Cada objeto debe tener dos claves: "type" y "content".
- "type" puede ser "normal" para texto regular, o "emphasis" para conceptos clave.
- "content" es el texto en sí.

EJEMPLO DE SALIDA OBLIGATORIA:
[
  {{ "type": "normal", "content": "El concepto principal es " }},
  {{ "type": "emphasis", "content": "la energía Crística" }},
  {{ "type": "normal", "content": ", que se menciona como el núcleo de la " }},
  {{ "type": "emphasis", "content": "evolución del alma" }},
  {{ "type": "normal", "content": ". (Fuente: archivo.srt, Timestamp: 00:01:23 --> 00:01:25)" }}
]
Esta regla no es negociable. Tu respuesta completa debe estar dentro de este formato JSON.

--- REGLA DE CITA ---
Incluye las citas de la fuente DENTRO del "content" de un objeto de tipo "normal", como se ve en el ejemplo. El formato es: `(Fuente: nombre_del_archivo.srt, Timestamp: HH:MM:SS --> HH:MM:SS)`.

Comienza tu labor, GERARD. Responde únicamente con el array JSON. No incluyas explicaciones adicionales fuera del JSON.
--- FIN DE INSTRUCCIONES DE PERSONALIDAD ---

Basándote ESTRICTAMENTE en las reglas y el contexto de abajo, responde la pregunta del usuario.

<contexto>
{context}
</contexto>

Pregunta del usuario: {input}
""")

# --- FUNCIÓN PARA FORMATEAR DOCUMENTOS (CON LIMPIEZA REFORZADA) ---
def get_cleaning_pattern():
    """Crea un patrón de regex robusto para eliminar textos no deseados."""
    texts_to_remove = [
        '[Spanish (auto-generated)]',
        '[DownSub.com]',
        '[Música]',
        '[Aplausos]'
    ]
    # Este patrón es más robusto: busca el texto dentro de los corchetes,
    # permitiendo espacios en blanco opcionales alrededor.
    robust_patterns = [r'\[\s*' + re.escape(text[1:-1]) + r'\s*\]' for text in texts_to_remove]
    return re.compile(r'|'.join(robust_patterns), re.IGNORECASE)

cleaning_pattern = get_cleaning_pattern()

def format_docs_with_metadata(docs):
    """Prepara los documentos recuperados, limpiando robustamente el contenido y los timestamps."""
    formatted_strings = []
    for doc in docs:
        source_filename = os.path.basename(doc.metadata.get('source', 'Fuente desconocida'))
        
        # 1. Limpieza de textos no deseados
        cleaned_content = cleaning_pattern.sub('', doc.page_content)

        # 2. ¡NUEVO! Eliminar milisegundos de los timestamps
        # El patrón busca HH:MM:SS,ms y lo reemplaza con HH:MM:SS
        cleaned_content = re.sub(r'(\d{2}:\d{2}:\d{2}),\d{3}', r'\1', cleaned_content)
        
        # 3. Limpieza de líneas vacías
        cleaned_content = "\n".join(line for line in cleaned_content.split('\n') if line.strip())
        
        if cleaned_content:
            formatted_strings.append(f"Fuente del Archivo: {source_filename}\nContenido:\n{cleaned_content}")
            
    return "\n\n---\n\n".join(formatted_strings)

# --- Cadena de recuperación (LCEL) ---
# ... el retrieval_chain se construye con `build_retrieval_chain(api_key)` cuando
# se ejecute el script como programa principal.

# --- NUEVA FUNCIÓN para convertir el JSON a texto plano para el log ---
def get_clean_text_from_json(json_string):
    """Convierte la respuesta JSON en una cadena de texto simple y legible."""
    try:
        match = re.search(r'\[.*\]', json_string, re.DOTALL)
        if not match:
            return json_string

        data = json.loads(match.group(0))
        full_text = "".join([item.get("content", "") for item in data])
        return full_text
    except Exception:
        return json_string

# --- NUEVA FUNCIÓN para guardar la conversación en un archivo ---
def save_to_log(question, user, answer_json):
    """Guarda la pregunta y la respuesta en un archivo de registro."""
    clean_answer = get_clean_text_from_json(answer_json)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:M:%S")
    
    with open("gerard_log.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Conversación del {timestamp} ---\n")
        f.write(f"Usuario: {user}\n")
        f.write(f"Pregunta: {question}\n")
        f.write(f"Respuesta de GERARD: {clean_answer}\n")
        f.write("="*40 + "\n\n")

# --- FUNCIÓN PARA IMPRIMIR LA RESPUESTA CON MÚLTIPLES COLORES ---
def print_json_answer(json_string):
    # Failsafe: volvemos a limpiar la respuesta final por si acaso
    cleaned_string = cleaning_pattern.sub('', json_string)
    
    try:
        match = re.search(r'\[.*\]', cleaned_string, re.DOTALL)
        if not match:
            print(f"{colorama.Fore.RED}Respuesta no es un JSON válido:\n{cleaned_string}")
            return

        data = json.loads(match.group(0))
        
        for item in data:
            content_type = item.get("type", "normal")
            content = item.get("content", "")
            
            if content_type == "emphasis":
                print(f"{colorama.Fore.YELLOW}{content}", end="")
            else:
                parts = re.split(r'(\(.*?\))', content)
                for part in parts:
                    if part.startswith('(') and part.endswith(')'):
                        print(f"{colorama.Fore.MAGENTA}{part}", end="")
                    else:
                        print(f"{colorama.Style.RESET_ALL}{part}", end="")
        print()
    except json.JSONDecodeError:
        print(f"{colorama.Fore.RED}Error: El modelo no devolvió un JSON válido. Respuesta recibida:\n{cleaned_string}")
    except Exception as e:
        print(f"{colorama.Fore.RED}Ocurrió un error inesperado al procesar la respuesta: {e}")

# --- Bucle de Interacción ---
def main():
    """Función principal que lanza el loop interactivo. Protegida para que no se ejecute al importar."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Modo rápido: evita cargar FAISS/LLM y usa una respuesta simulada")
    args = parser.parse_args()

    load_dotenv()

    # Fast mode: no inicializar LLM/FAISS (útil para pruebas rápidas)
    if args.fast:
        class DummyChain:
            def invoke(self, payload):
                # Respuesta de ejemplo respetando el formato requerido
                return json.dumps([
                    {"type": "normal", "content": "[Modo rápido] Respuesta simulada para la pregunta: " + payload.get('input', '')}
                ])

        retrieval_chain = DummyChain()
    else:
        # Intentar obtener la key desde keyring o entornos
        api_key = get_api_key()
        if not api_key:
            print("Error: No se encontró la API key. Guarda la clave en el keyring (scripts/store_key_keyring.py), en la variable de entorno GOOGLE_API_KEY, o en .env")
            return

        retrieval_chain = build_retrieval_chain(api_key)

    print("GERARD listo. Escribe tu pregunta o 'salir' para terminar.")
    user_name = input("Por favor, introduce tu nombre para comenzar: ")

    while True:
        prompt_text = f"\nTu pregunta {colorama.Fore.BLUE}{user_name.upper()}{colorama.Style.RESET_ALL}: "
        pregunta = input(prompt_text)

        if pregunta.lower() == 'salir':
            break

        print("Buscando...")
        try:
            answer = retrieval_chain.invoke({"input": pregunta})
            print("\nRespuesta de GERARD:")
            print_json_answer(answer)
            save_to_log(pregunta, user_name.upper(), answer)

        except Exception as e:
            print(f"\n{colorama.Fore.RED}Ocurrió un error al procesar tu pregunta: {e}")


if __name__ == "__main__":
    main()


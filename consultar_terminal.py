import os
import json
import re
import colorama
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# Inicializamos colorama para que los colores funcionen en todas las terminales
colorama.init(autoreset=True)

# --- Carga la API Key ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
    exit()

# --- Usamos el modelo 'pro' que es mejor para seguir instrucciones complejas como JSON ---
api_key = os.environ.get("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="models/gemini-2.5-pro", google_api_key=api_key)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key),
)

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
    except:
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
print("GERARD listo. Escribe tu pregunta o 'salir' para terminar.")

# --- ¡NUEVO! Pedir el nombre del usuario una sola vez al inicio ---
user_name = input("Por favor, introduce tu nombre para comenzar: ")

while True:
    # --- ¡MODIFICADO! Usar el nombre personalizado en el prompt ---
    prompt_text = f"\nTu pregunta {colorama.Fore.BLUE}{user_name.upper()}{colorama.Style.RESET_ALL}: "
    pregunta = input(prompt_text)

    if pregunta.lower() == 'salir':
        break

    print("Buscando...")
    try:
        answer = retrieval_chain.invoke({"input": pregunta})
        print("\nRespuesta de GERARD:")
        print_json_answer(answer)
        
        # Guardamos el registro con el nombre del usuario
        save_to_log(pregunta, user_name.upper(), answer)

    except Exception as e:
        print(f"\n{colorama.Fore.RED}Ocurrió un error al procesar tu pregunta: {e}")


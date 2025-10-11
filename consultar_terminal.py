import os
import json
import re
import colorama
import argparse
import threading
import itertools
import sys
import time
import getpass
import uuid
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
prompt = ChatPromptTemplate.from_template(r"""
🔬 GERARD v3.01 - Sistema de Análisis Investigativo Avanzado
IDENTIDAD DEL SISTEMA
═══════════════════════════════════════════════════════════
Nombre: GERARD
Versión: 3.01 - Analista Forense Documental
Modelo Base: Gemini Pro Latest 2.5
Temperatura: 0.2-0.3 (Máxima Precisión y Consistencia)
Especialización: Criptoanálisis de Archivos .srt
═══════════════════════════════════════════════════════════
MISIÓN CRÍTICA
Eres GERARD, un sistema de inteligencia analítica especializado en arqueología documental de archivos de subtítulos (.srt). Tu propósito es descubrir patrones ocultos, mensajes encriptados y conexiones invisibles que emergen al correlacionar múltiples documentos mediante técnicas forenses avanzadas.
Configuración de Temperatura Optimizada (0.2-0.3)
Esta temperatura baja garantiza:
• Consistencia absoluta entre consultas repetidas
• Reproducibilidad de hallazgos para verificación
• Precisión quirúrgica en extracción de datos
• Eliminación de variabilidad en respuestas críticas
• Confiabilidad forense en análisis investigativos
________________________________________
🚨 PROTOCOLOS DE SEGURIDAD ANALÍTICA
REGLAS ABSOLUTAS (Nivel de Cumplimiento: 100%)
🔴 PROHIBICIÓN NIVEL 1: FABRICACIÓN DE DATOS
├─ ❌ NO inventar información bajo ninguna circunstancia
├─ ❌ NO usar conocimiento del modelo base (entrenamiento general)
├─ ❌ NO suponer o inferir más allá de lo textualmente disponible
└─ ❌ NO completar información faltante con lógica externa

🔴 PROHIBICIÓN NIVEL 2: CONTAMINACIÓN ANALÍTICA
├─ ❌ NO mezclar análisis con citas textuales
├─ ❌ NO parafrasear cuando se requiere texto literal
├─ ❌ NO interpretar sin declarar explícitamente que es interpretación
└─ ❌ NO omitir información contradictoria si existe

🟢 MANDATOS OBLIGATORIOS
├─ ✅ Cada afirmación DEBE tener cita textual verificable
├─ ✅ Cada cita DEBE incluir: [Documento] + [Timestamp] + [Texto Literal]
├─ ✅ Cada análisis DEBE separarse claramente de evidencias
├─ ✅ Cada consulta DEBE ejecutar los 8 Protocolos de Búsqueda Profunda
└─ ✅ Cada respuesta DEBE incluir nivel de confianza estadístico
________________________________________
🔍 SISTEMA DE ANÁLISIS MULTINIVEL
NIVEL 1: EXTRACCIÓN SUPERFICIAL (Baseline)
Objetivo: Captura literal de información explícita
Técnica: Lectura directa y indexación
Profundidad: 0-20% del contenido oculto
NIVEL 2: ANÁLISIS CORRELACIONAL (Intermediate)
Objetivo: Conexión de fragmentos dispersos
Técnicas:
    ├─ Mapeo de relaciones temáticas
    ├─ Detección de patrones recurrentes
    ├─ Identificación de complementariedades
    ├─ Triangulación de fuentes múltiples
    └─ Construcción de narrativas coherentes
Profundidad: 20-50% del contenido oculto
NIVEL 3: CRIPTOANÁLISIS FORENSE (Advanced)
Objetivo: Descubrimiento de mensajes encriptados
Profundidad: 50-85% del contenido oculto
________________________________________
🔐 PROTOCOLOS DE BÚSQUEDA PROFUNDA (8 CHECKS OBLIGATORIOS)
CHECK #1: ANÁLISIS ACRÓSTICO MULTINIVEL
MÉTODO: ... (ejecutar los pasos descritos en el protocolo suministrado)
CHECK #2: ANÁLISIS DE PATRONES NUMÉRICOS
MÉTODO: ...
CHECK #3: ANÁLISIS DE PALABRAS CLAVE DISTRIBUIDAS
MÉTODO: ...
CHECK #4: ANÁLISIS SECUENCIAL CRONOLÓGICO
MÉTODO: ...
CHECK #5: ANÁLISIS CONTEXTUAL DE FRAGMENTACIÓN
MÉTODO: ...
CHECK #6: ANÁLISIS DE ANOMALÍAS Y REPETICIONES
MÉTODO: ...
CHECK #7: ANÁLISIS DE OMISIONES DELIBERADAS
MÉTODO: ...
CHECK #8: ANÁLISIS DE METADATOS Y MARCADORES OCULTOS
MÉTODO: ...
________________________________________
📋 ESTRUCTURA DE RESPUESTA OPTIMIZADA PARA TEMP 0.2-0.3
FORMATO ESTANDARIZADO (Reproducibilidad Garantizada)
═══════════════════════════════════════════════════════════
🔬 ANÁLISIS
═══════════════════════════════════════════════════════════
Timestamp de Análisis: [{date}]
Consulta Procesada: "{input}"
Temperatura Operativa: 0.2-0.3
Hash de Sesión: [{session_hash}]
═══════════════════════════════════════════════════════════
SECCIÓN 1: SÍNTESIS INVESTIGATIVA
[Resuma hallazgos y evidencias; siga estrictamente las reglas de cita y separación de evidencia/interpretación]
SECCIÓN 2: EVIDENCIA FORENSE ESTRUCTURADA
[Agrupe por documento y cite por timestamp: siempre texto literal]
SECCIÓN 3: ÍNDICE DE FUENTES Y MAPEO
[Reporte de cobertura y relevancia]
SECCIÓN 4: METADATOS Y GARANTÍA DE CALIDAD
[Reporte de ejecución de checks y nivel de confianza]
═══════════════════════════════════════════════════════════
FIN DEL ANÁLISIS

Basándote estrictamente en el contenido disponible en el contexto (no accedas a fuentes externas), responde la consulta del usuario respetando todas las prohibiciones y mandatos arriba definidos.
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
        
        # Eliminar extensión .srt para fuentes más limpias
        if source_filename.endswith('.srt'):
            source_filename = source_filename[:-4]
        
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
    parser.add_argument("--no-store", action="store_true", help="No almacenar la API key en keyring aunque se provea interactiva")
    args = parser.parse_args()

    load_dotenv()

    # Intentar obtener la key desde keyring o entornos
    api_key = get_api_key()
    if not api_key:
        # Pedir interactivamente la clave al usuario
        print("No se encontró GOOGLE_API_KEY en keyring/entorno. Puedes pegar tu clave ahora (se ocultará).")
        try:
            entered = getpass.getpass(prompt="Introduce tu GOOGLE_API_KEY (vacío para cancelar): ")
        except Exception:
            entered = input("Introduce tu GOOGLE_API_KEY (vacío para cancelar): ")

        if not entered:
            print("No se proporcionó clave. Abortando.")
            return

        api_key = entered.strip()
        # Ofrecer guardar en keyring
        if not args.no_store:
            try:
                save_choice = input("¿Deseas guardar esta clave en el keyring del sistema para futuras ejecuciones? [y/N]: ")
                if save_choice.lower().startswith('y'):
                    try:
                        keyring.set_password('consultor-gerard', 'google_api_key', api_key)
                        print("Clave guardada en keyring con servicio 'consultor-gerard'.")
                    except Exception as e:
                        print(f"No se pudo guardar en keyring: {e}")
            except Exception:
                # No crítico si falla la interacción para guardar
                pass

    # Construir la cadena de recuperación real
    try:
        retrieval_chain = build_retrieval_chain(api_key)
    except Exception as e:
        print(f"Error inicializando el pipeline real: {e}")
        return

    print("GERARD listo. Escribe tu pregunta o 'salir' para terminar.")
    user_name = input("Por favor, introduce tu nombre para comenzar: ")

    while True:
        prompt_text = f"\nTu pregunta {colorama.Fore.BLUE}{user_name.upper()}{colorama.Style.RESET_ALL}: "
        pregunta = input(prompt_text)

        if pregunta.lower() == 'salir':
            break

        print("Buscando...")
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session_hash = str(uuid.uuid4())
            answer = retrieval_chain.invoke({"input": pregunta, "date": ts, "session_hash": session_hash})
            print("\nRespuesta de GERARD:")
            print_json_answer(answer)
            save_to_log(pregunta, user_name.upper(), answer)

        except Exception as e:
            print(f"\n{colorama.Fore.RED}Ocurrió un error al procesar tu pregunta: {e}")


if __name__ == "__main__":
    main()


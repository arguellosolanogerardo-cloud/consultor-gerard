"""
Este script permite interactuar con el Consultor Gerard a través de la terminal.

Carga una base de datos vectorial FAISS y utiliza un modelo de lenguaje de Google (Gemini)
para responder preguntas. La respuesta se estructura en un resumen seguido de una lista
de las citas textuales más relevantes que se usaron para generar el resumen.

Funcionalidades:
- Carga de variables de entorno para la clave de API.
- Carga el índice FAISS local.
- Configuración de un modelo de chat que genera un resumen.
- Mantiene un historial de la conversación.
- Bucle interactivo para que el usuario haga preguntas.
- Lógica de post-procesamiento para mostrar los documentos fuente directamente como citas.

Uso:
- Ejecuta el script con `python consultar_terminal.py`.
"""

import os
import re
import gender_guesser.detector as gender
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Importar sistema de logging
from interaction_logger import InteractionLogger

# Inicializar colorama para que funcione en Windows
init()

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print(Fore.RED + "No se encontró la clave de API para Google. Por favor, configura la variable de entorno GOOGLE_API_KEY." + Style.RESET_ALL)
    exit()

# Ruta al índice FAISS
FAISS_INDEX_PATH = "faiss_index"

# Inicializar detector de género
gender_detector = gender.Detector()

def detect_gender(name):
    """
    Detecta el género de un nombre usando gender-guesser y una lista de nombres comunes en español.
    Retorna 'male' o 'female', por defecto 'male' si no se puede determinar.
    """
    # Tomar solo el primer nombre y convertir a minúsculas para comparación
    first_name = name.strip().split()[0] if name.strip() else name
    first_name_lower = first_name.lower()
    
    # Listas de nombres femeninos y masculinos comunes en español
    female_names = {
        'maria', 'ana', 'rosa', 'carmen', 'laura', 'marta', 'elena', 'isabel', 
        'cristina', 'patricia', 'lucia', 'sara', 'paula', 'claudia', 'beatriz',
        'silvia', 'pilar', 'raquel', 'monica', 'angela', 'teresa', 'lorena',
        'natalia', 'veronica', 'susana', 'alicia', 'rocio', 'yolanda', 'gloria',
        'mercedes', 'julia', 'carolina', 'daniela', 'andrea', 'valeria', 'camila',
        'sofia', 'valentina', 'isabella', 'gabriela', 'mariana', 'alejandra',
        'fernanda', 'paola', 'carolina', 'adriana', 'marcela', 'diana', 'sandra',
        'jessica', 'karen', 'vanessa', 'stephanie', 'katherine', 'nicole', 'emily',
        'ashley', 'michelle', 'brittany', 'amber', 'crystal', 'melissa', 'rebecca',
        'martha', 'ruth', 'esther', 'judith', 'deborah', 'sarah', 'hannah', 'rachel',
        'leah', 'anna', 'elizabeth', 'mary', 'linda', 'barbara', 'susan', 'karen',
        'nancy', 'betty', 'helen', 'dorothy', 'sandra', 'ashley', 'kimberly',
        'donna', 'emily', 'carol', 'michelle', 'amanda', 'melissa', 'deborah'
    }
    
    male_names = {
        'juan', 'jose', 'antonio', 'manuel', 'francisco', 'david', 'carlos',
        'miguel', 'javier', 'pedro', 'jesus', 'alejandro', 'fernando', 'sergio',
        'luis', 'pablo', 'jorge', 'alberto', 'rafael', 'daniel', 'andres',
        'roberto', 'ricardo', 'eduardo', 'enrique', 'angel', 'ramon', 'vicente',
        'raul', 'oscar', 'jaime', 'ignacio', 'diego', 'adrian', 'ivan', 'ruben',
        'alvaro', 'marcos', 'cesar', 'guillermo', 'alfredo', 'santiago', 'martin',
        'nicolas', 'sebastian', 'mateo', 'benjamin', 'samuel', 'gabriel', 'leonardo',
        'john', 'michael', 'david', 'james', 'robert', 'william', 'richard',
        'joseph', 'thomas', 'charles', 'christopher', 'daniel', 'matthew', 'anthony',
        'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua', 'kenneth', 'kevin',
        'brian', 'george', 'timothy', 'ronald', 'edward', 'jason', 'jeffrey'
    }
    
    # Primero verificar en las listas personalizadas
    if first_name_lower in female_names:
        return 'female'
    elif first_name_lower in male_names:
        return 'male'
    
    # Si no está en las listas, usar gender-guesser
    detected = gender_detector.get_gender(first_name)
    
    # gender-guesser retorna: 'male', 'female', 'mostly_male', 'mostly_female', 'andy' (andrógino), 'unknown'
    if detected in ['female', 'mostly_female']:
        return 'female'
    else:
        return 'male'  # Por defecto masculino para casos ambiguos

def get_conversational_chain():
    """
    Configura y retorna la cadena de conversación y recuperación (RAG).
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro-latest",
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            max_output_tokens=8192
        )

        # Nuevo prompt para GERARD 3.0
        prompt_template = '''IDENTIDAD DEL SISTEMA
Nombre: GERARD
Versión: 3.0 - Analista Investigativo

INSTRUCCIÓN CRÍTICA SOBRE FORMATO DE REFERENCIAS:
Cuando cites información de los documentos, DEBES usar el siguiente formato EXACTO para las referencias:

(Nombre del archivo .srt - MM:SS)

Ejemplo correcto: (MEDITACION 107 LA CURA MILAGROSA MAESTRO ALANISO - 00:46)
Ejemplo INCORRECTO: (00:00:46,840)

SIEMPRE incluye:
1. El nombre completo del archivo fuente (sin prefijos como [Spanish (auto-generated)])
2. Un guión separador " - "
3. El timestamp en formato MM:SS (sin milisegundos)

DOCUMENTOS DISPONIBLES:
{context}

CONSULTA DEL USUARIO:
{question}

RECUERDA: Cada vez que cites información, usa el formato (Nombre archivo - MM:SS)
'''
        
        QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return chain
    except Exception as e:
        print(Fore.RED + f"Error al cargar la cadena de conversación: {e}" + Style.RESET_ALL)
        return None

def clean_source_name(source_name):
    """Limpia el nombre de un archivo fuente de textos repetitivos."""
    prefixes_to_remove = ["[Spanish (auto-generated)]", "[Spanish (Latin America)]", "[Spanish]"]
    for prefix in prefixes_to_remove:
        if source_name.startswith(prefix):
            source_name = source_name[len(prefix):].strip()
    
    if source_name.endswith("[DownSub.com].srt"):
        source_name = source_name[:-len("[DownSub.com].srt")].strip()
    
    return source_name

def clean_srt_content(text):
    """Elimina los números de secuencia y timestamps de un fragmento de SRT."""
    # Eliminar timestamps (e.g., 00:00:20,000 --> 00:00:25,000)
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}', '', text)
    # Eliminar números de secuencia al inicio de las líneas
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    # Eliminar líneas vacías resultantes
    return '\n'.join(line for line in text.split('\n') if line.strip()).strip()

def main():
    """
    Función principal que maneja el bucle de interacción con el usuario.
    """
    chain = get_conversational_chain()
    if not chain:
        return

    # Inicializar logger para terminal
    logger = InteractionLogger(platform="terminal", anonymize=False)
    
    print(Fore.CYAN + "GERARD listo. Escribe tu pregunta o 'salir' para terminar." + Style.RESET_ALL)
    
    user_name = input("Por favor, introduce tu nombre para comenzar: ")
    user_gender = detect_gender(user_name)
    gender_suffix = "A" if user_gender == 'female' else "O"
    print(Fore.GREEN + f"\nBIENVENID{gender_suffix} {user_name.upper()}! Puedes empezar a preguntar." + Style.RESET_ALL)

    while True:
        user_question = input(f"\nTu pregunta {user_name.upper()}: ")

        if user_question.lower() == 'salir':
            print(Fore.YELLOW + "Gracias por conversar. ¡Hasta luego!" + Style.RESET_ALL)
            break
        
        if user_question:
            session_id = None
            try:
                print(Fore.YELLOW + "Buscando..." + Style.RESET_ALL)
                
                # Iniciar logging de la interacción
                session_id = logger.start_interaction(
                    user=user_name,
                    question=user_question
                )
                
                # Marcar inicio de consulta RAG
                logger.mark_phase(session_id, "rag_start")
                
                # Marcar inicio de consulta LLM
                logger.mark_phase(session_id, "llm_start")
                
                result = chain.invoke({"question": user_question})
                
                # Marcar fin de consulta LLM
                logger.mark_phase(session_id, "llm_end")
                
                answer = result["answer"]
                sources = result.get("source_documents", [])
                
                # Registrar respuesta
                logger.log_response(session_id, answer, sources)
                
                # Marcar inicio de procesamiento
                logger.mark_phase(session_id, "processing_start")

                print(Fore.GREEN + "\nRespuesta de GERARD:" + Style.RESET_ALL)
                
                # Aplicar color violeta al texto dentro de paréntesis
                formatted_answer = re.sub(
                    r'\(([^)]+)\)',
                    lambda m: f"{Fore.MAGENTA}({m.group(1)}){Style.RESET_ALL}",
                    answer
                )
                print(formatted_answer) # Imprime el resumen con formato

                # Mostrar las citas textuales directamente de los documentos fuente
                if sources:
                    print(Fore.CYAN + "\n--- Citas Textuales ---" + Style.RESET_ALL)
                    
                    for doc in sources:
                        # Extraer la información de la fuente
                        source_path = doc.metadata.get("source", "Fuente desconocida")
                        source_file = os.path.basename(source_path)
                        cleaned_name = clean_source_name(source_file)
                        
                        # Extraer y formatear el timestamp
                        timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})', doc.page_content)
                        if timestamp_match:
                            timestamp = timestamp_match.group(0)
                            short_timestamp = re.sub(r',\d{3}', '', timestamp)
                            source_info = f"(Fuente: {cleaned_name}, Timestamp: {short_timestamp})"
                        else:
                            source_info = f"(Fuente: {cleaned_name}, Timestamp: No disponible)"

                        # Limpiar el contenido del SRT para mostrar solo el texto
                        quote_text = clean_srt_content(doc.page_content)

                        # Formatear la salida
                        highlighted_quote = f"{Fore.YELLOW}{quote_text}{Style.RESET_ALL}"
                        violet_source = f" {Fore.MAGENTA}{source_info}{Style.RESET_ALL}"
                        
                        print(f"- {highlighted_quote}{violet_source}\n")
                
                # Marcar fin de procesamiento
                logger.mark_phase(session_id, "processing_end")
                
                # Finalizar logging con éxito
                if session_id:
                    logger.end_interaction(session_id, status="success")

            except Exception as e:
                print(Fore.RED + f"Ocurrió un error al procesar tu pregunta: {e}" + Style.RESET_ALL)
                
                # Finalizar logging con error
                if session_id:
                    logger.end_interaction(session_id, status="error", error=str(e))

if __name__ == "__main__":
    main()

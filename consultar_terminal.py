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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from colorama import Fore, Style, init

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

def get_conversational_chain():
    """
    Configura y retorna la cadena de conversación y recuperación (RAG).
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0.7)

        # Un prompt más simple que se enfoca en generar un buen resumen.
        prompt_template = """Eres un asistente servicial llamado GERARD. Tu tarea es responder la pregunta del usuario de forma coherente y útil, creando un resumen basado únicamente en el siguiente contexto. No inventes información. Si no sabes la respuesta, di que no has encontrado información suficiente.

        Contexto:
        {context}

        Pregunta: {question}

        Respuesta de GERARD:"""
        
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

    print(Fore.CYAN + "GERARD listo. Escribe tu pregunta o 'salir' para terminar." + Style.RESET_ALL)
    
    user_name = input("Por favor, introduce tu nombre para comenzar: ")
    print(Fore.GREEN + f"\n¡Hola, {user_name}! Puedes empezar a preguntar." + Style.RESET_ALL)

    while True:
        user_question = input(f"\nTu pregunta {user_name.upper()}: ")

        if user_question.lower() == 'salir':
            print(Fore.YELLOW + "Gracias por conversar. ¡Hasta luego!" + Style.RESET_ALL)
            break
        
        if user_question:
            try:
                print(Fore.YELLOW + "Buscando..." + Style.RESET_ALL)
                
                result = chain.invoke({"question": user_question})
                answer = result["answer"]
                sources = result.get("source_documents", [])

                print(Fore.GREEN + "\nRespuesta de GERARD:" + Style.RESET_ALL)
                print(answer) # Imprime el resumen

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

            except Exception as e:
                print(Fore.RED + f"Ocurrió un error al procesar tu pregunta: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()


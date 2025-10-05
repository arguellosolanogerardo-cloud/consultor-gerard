"""
Este script crea una aplicación web utilizando Streamlit para interactuar con el Consultor Gerard.

La respuesta se estructura en un resumen seguido de una lista de las citas textuales
más relevantes que se usaron para generar el resumen, mostrándolas directamente.

Funcionalidades:
- Interfaz web creada con Streamlit.
- Carga de variables de entorno para la clave de API.
- Carga de una base de datos vectorial FAISS pre-construida.
- Configuración de un modelo de chat que genera un resumen.
- Gestión del estado de la sesión para mantener el historial de chat.
- Lógica de post-procesamiento para mostrar los documentos fuente directamente como citas.
- Opciones de exportación a .txt y .pdf.
- Enlaces para compartir la respuesta por Email, WhatsApp y Telegram.

Uso:
- Ejecuta la aplicación con `streamlit run consultar_web.py`.
"""

import streamlit as st
import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from urllib.parse import quote

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("No se encontró la clave de API para Google. Por favor, configura la variable de entorno GOOGLE_API_KEY en los secretos de Streamlit.")
    st.stop()

# Ruta al índice FAISS
FAISS_INDEX_PATH = "faiss_index"

@st.cache_resource
def get_conversational_chain():
    """
    Carga y configura la cadena de conversación y recuperación (RAG).
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
        
        if not os.path.exists(FAISS_INDEX_PATH):
            st.error(f"La base de datos vectorial no se encuentra en la ruta: {FAISS_INDEX_PATH}. Asegúrate de que la carpeta existe.")
            st.stop()

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
        st.error(f"Ocurrió un error al cargar el modelo o la base de datos: {e}")
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
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}', '', text)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    return '\n'.join(line for line in text.split('\n') if line.strip()).strip()





def main():
    st.set_page_config(page_title="Consultor Gerard", page_icon="🤖")
    st.title("🤖 Consultor Gerard")
    st.write("Bienvenido. Soy Gerard, tu asistente virtual. Estoy aquí para responder tus preguntas basado en un conjunto de documentos.")

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""

    if st.session_state.conversation_chain is None:
        with st.spinner("Cargando a Gerard... Por favor, espera."):
            st.session_state.conversation_chain = get_conversational_chain()
        if st.session_state.conversation_chain is None:
            st.stop()

    if not st.session_state.user_name:
        st.session_state.user_name = st.text_input("Por favor, introduce tu nombre para comenzar:")
        if st.session_state.user_name:
            st.rerun()
    else:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        user_question = st.chat_input(f"Hola {st.session_state.user_name}, ¿en qué puedo ayudarte?")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        result = st.session_state.conversation_chain.invoke({"question": user_question})
                        answer = result["answer"]
                        sources = result.get("source_documents", [])
                        
                        final_answer_html = f"<p>{answer}</p>" # Empezar con el resumen

                        if sources:
                            final_answer_html += "<h3>--- Citas Textuales ---</h3>"
                            quotes_html_list = "<ul>"
                            
                            for doc in sources:
                                source_path = doc.metadata.get("source", "Fuente desconocida")
                                source_file = os.path.basename(source_path)
                                cleaned_name = clean_source_name(source_file)
                                
                                timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})', doc.page_content)
                                if timestamp_match:
                                    timestamp = timestamp_match.group(0)
                                    short_timestamp = re.sub(r',\d{3}', '', timestamp)
                                    source_info = f"(Fuente: {cleaned_name}, Timestamp: {short_timestamp})"
                                else:
                                    source_info = f"(Fuente: {cleaned_name}, Timestamp: No disponible)"

                                quote_text = clean_srt_content(doc.page_content)

                                highlighted_quote = f'<span style="color: yellow;">{quote_text}</span>'
                                violet_source = f' <span style="color: violet;">{source_info}</span>'
                                quotes_html_list += f"<li>{highlighted_quote}{violet_source}</li>"

                            quotes_html_list += "</ul>"
                            final_answer_html += quotes_html_list
                        
                        st.markdown(final_answer_html, unsafe_allow_html=True)
                        st.session_state.chat_history.append({"role": "assistant", "content": final_answer_html})

                        # --- Funcionalidad de Exportar y Compartir ---
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("📥 Exportar")
                            st.download_button(
                                label="Descargar como HTML",
                                data=final_answer_html.encode('utf-8'),
                                file_name="respuesta_gerard.html",
                                mime="text/html"
                            )
                        
                        with col2:
                            st.subheader("🔗 Compartir")
                            # Para compartir, usamos una versión de texto plano simple
                            plain_text_for_sharing = re.sub('<[^<]+?>', '', final_answer_html)
                            encoded_text = quote(plain_text_for_sharing)
                            st.markdown(f"[Compartir por Email](mailto:?subject=Respuesta%20de%20Gerard&body={encoded_text})")
                            st.markdown(f"[Compartir en WhatsApp](https://api.whatsapp.com/send?text={encoded_text})")
                            st.markdown(f"[Compartir en Telegram](https://t.me/share/url?url=&text={encoded_text})")

                    except Exception as e:
                        st.error(f"Ocurrió un error al procesar tu pregunta: {e}")

if __name__ == "__main__":
    main()


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
- Sistema de logging completo de interacciones.

Uso:
- Ejecuta la aplicación con `streamlit run consultar_web.py`.
"""

import streamlit as st
import os
import re
import base64
import gender_guesser.detector as gender
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime

from urllib.parse import quote
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from io import BytesIO

# Importar sistema de logging
from interaction_logger import InteractionLogger

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("No se encontró la clave de API para Google. Por favor, configura la variable de entorno GOOGLE_API_KEY en los secretos de Streamlit.")
    st.stop()

# Ruta al índice FAISS
FAISS_INDEX_PATH = "faiss_index"

# Inicializar detector de género
gender_detector = gender.Detector()

def load_gif_as_base64(gif_path):
    """Carga un GIF y lo convierte a base64 para usarlo en HTML."""
    try:
        with open(gif_path, "rb") as f:
            gif_data = f.read()
        return base64.b64encode(gif_data).decode()
    except Exception as e:
        st.error(f"Error al cargar el GIF: {e}")
        return None

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
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro-latest",
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            max_output_tokens=8192
        )
        
        # Prompt para GERARD 3.0
        prompt_template = '''[El prompt completo de GERARD se mantiene igual...]

DOCUMENTOS DISPONIBLES:
{context}

CONSULTA DEL USUARIO:
{question}
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

def export_to_txt(html_content, user_question):
    """Convierte HTML a texto plano para exportación TXT."""
    # Remover tags HTML
    plain_text = re.sub('<[^<]+?>', '', html_content)
    # Agregar encabezado
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"GERARD - Respuesta\nFecha: {timestamp}\nPregunta: {user_question}\n\n{'='*60}\n\n"
    return header + plain_text

def export_to_markdown(html_content, user_question):
    """Convierte HTML a Markdown para exportación."""
    # Agregar encabezado
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown = f"# GERARD - Respuesta\n\n"
    markdown += f"**Fecha:** {timestamp}  \n"
    markdown += f"**Pregunta:** {user_question}\n\n"
    markdown += "---\n\n"
    
    # Convertir HTML básico a Markdown
    content = html_content
    # Convertir <h3> a ###
    content = re.sub(r'<h3>(.*?)</h3>', r'### \1', content)
    # Convertir <p> simplemente removiendo tags
    content = re.sub(r'<p>(.*?)</p>', r'\1\n\n', content)
    # Convertir <ul> y <li>
    content = re.sub(r'<ul>', '', content)
    content = re.sub(r'</ul>', '\n', content)
    content = re.sub(r'<li>', '- ', content)
    content = re.sub(r'</li>', '\n', content)
    # Remover spans y otros tags
    content = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', content)
    content = re.sub(r'<[^<]+?>', '', content)
    
    markdown += content
    return markdown

def export_to_pdf(html_content, user_question):
    """Convierte HTML a PDF para exportación."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Contenedor para los elementos del PDF
    elements = []
    
    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='purple',
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='darkblue',
        spaceAfter=12
    )
    
    normal_style = styles['BodyText']
    normal_style.fontSize = 11
    normal_style.leading = 14
    
    # Título
    elements.append(Paragraph("GERARD - Respuesta", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"<b>Fecha:</b> {timestamp}", normal_style))
    elements.append(Paragraph(f"<b>Pregunta:</b> {user_question}", normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Línea separadora
    elements.append(Paragraph("<hr/>", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Convertir HTML a texto con formato simple para PDF
    content = html_content
    
    # Procesar h3 tags
    h3_pattern = re.compile(r'<h3>(.*?)</h3>', re.DOTALL)
    for match in h3_pattern.finditer(content):
        heading_text = match.group(1)
        content = content.replace(match.group(0), f'|||HEADING|||{heading_text}|||HEADING|||')
    
    # Procesar listas
    content = re.sub(r'<ul>', '', content)
    content = re.sub(r'</ul>', '', content)
    content = re.sub(r'<li>', '• ', content)
    content = re.sub(r'</li>', '<br/>', content)
    
    # Remover otros tags HTML pero mantener <br/>
    content = re.sub(r'<(?!br/>)[^<]+?>', '', content)
    
    # Dividir por secciones
    parts = content.split('|||HEADING|||')
    
    for i, part in enumerate(parts):
        if part.strip():
            if i > 0 and i % 2 == 1:  # Es un heading
                elements.append(Spacer(1, 0.2*inch))
                elements.append(Paragraph(part.strip(), heading_style))
            else:  # Es contenido normal
                # Dividir por párrafos
                paragraphs = part.split('<br/>')
                for para in paragraphs:
                    if para.strip():
                        elements.append(Paragraph(para.strip(), normal_style))
                        elements.append(Spacer(1, 0.1*inch))
    
    # Construir PDF
    doc.build(elements)
    
    buffer.seek(0)
    return buffer.getvalue()


def main():
    st.set_page_config(page_title="GERARD", page_icon="🔮")
    st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Orbitron&display=swap');

@keyframes pulse {
  0% {
    transform: scale(1);
    text-shadow: 0 0 3px #c07dfc, 0 0 5px #c07dfc;
  }
  50% {
    transform: scale(1.05);
    text-shadow: 0 0 8px #c07dfc, 0 0 12px #c07dfc;
  }
  100% {
    transform: scale(1);
    text-shadow: 0 0 3px #c07dfc, 0 0 5px #c07dfc;
  }
}

@keyframes rotate {
  0% {
    transform: rotate(0deg);
  }
  25% {
    transform: rotate(-15deg);
  }
  50% {
    transform: rotate(15deg);
  }
  75% {
    transform: rotate(-15deg);
  }
  100% {
    transform: rotate(0deg);
  }
}

@keyframes pulsateRed {
  0% {
    transform: scale(1);
    color: red;
    text-shadow: 0 0 5px red;
  }
  50% {
    transform: scale(1.3);
    color: #ff4444;
    text-shadow: 0 0 15px red, 0 0 25px red;
  }
  100% {
    transform: scale(1);
    color: red;
    text-shadow: 0 0 5px red;
  }
}

@keyframes pulse-green {
  0% {
    transform: scale(1);
    text-shadow: 0 0 5px #00ff00, 0 0 10px #00ff00;
  }
  50% {
    transform: scale(1.1);
    text-shadow: 0 0 15px #00ff00, 0 0 25px #00ff00;
  }
  100% {
    transform: scale(1);
    text-shadow: 0 0 5px #00ff00, 0 0 10px #00ff00;
  }
}

.nombre-label {
  font-family: 'Orbitron', sans-serif;
  font-size: 3rem;
  color: #00ff00;
  text-align: center;
  animation: pulse-green 1.5s infinite;
  margin-bottom: 10px;
}

.question-mark-start {
  display: inline-block;
  font-size: 1.5rem;
  color: #FFD700;
  margin-right: 10px;
  animation: rotate 1.5s infinite;
}

.question-text {
  color: orange;
  font-weight: bold;
  text-transform: uppercase;
}

.question-mark-end {
  display: inline-block;
  font-size: 1.5rem;
  margin-left: 5px;
  animation: pulsateRed 1s infinite;
}

.title-gerard {
  font-family: 'Audiowide', cursive;
  font-size: 8rem;
  font-weight: 400;
  color: #9d4edd;
  text-align: center;
  animation: pulse 2s infinite;
  margin-bottom: 20px;
  letter-spacing: 0.5rem;
}

.welcome-message {
  font-family: 'Orbitron', sans-serif;
  font-size: 2.5rem;
  font-weight: 700;
  color: #00ff00;
  text-align: center;
  margin: 30px 0;
  text-shadow: 0 0 10px #00ff00;
}

.red-label {
  color: red !important;
  font-weight: bold;
  font-size: 1.1rem;
}

.centered-bigger-red-label {
  color: red !important;
  font-weight: bold;
  font-size: 1.5rem; /* Increased from 1.1rem */
  text-align: center;
}

/* Estilo para el input de texto */
div[data-testid="stTextInput"] label {
  color: red !important;
  font-weight: bold;
  font-size: 1.1rem;
}
</style>

<div class="title-gerard">GERARD</div>
''', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align: center; color: lightblue;">
    HOLA. TE AYUDARE A ENCONTRAR EL TITULO, MINUTO Y SEGUNDO SOBRE LAS PREGUNTAS QUE REALIZES ACERCA DE LAS MEDITACIONES/MENSAJES DE LOS 9 MAESTROS: ALANISO,AXEL,ADIEL,AZOES,AVIATAR,ALADIM,ALIESTRO,ALAN,AZEN,EL PADRE AMOR, EL GRAN MAESTRO JESUS Y LA GRAN MADRE.                                                DE MENSAJES CANALIZADOS POR SARITA OTERO QUE YA HAYAS ESCUCHADO PERO QUE NO RECUERDAS EL MINUTO EXACTO DE ALGUN MENSAJE O MENSAJES EN ESPECIAL QUE TENGAS EN MENTE. PARA QUE ASI ESCUCHES DE NUEVO. ANALIZES Y DISCIERNAS MEJOR LOS MENSAJES DENTRO DE LOS MENSAJES DE EL CONOCIMIENTO UNIVERSAL.
</div>
""", unsafe_allow_html=True)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_gender" not in st.session_state:
        st.session_state.user_gender = ""
    if "logger" not in st.session_state:
        # Inicializar logger para web
        st.session_state.logger = InteractionLogger(platform="web", anonymize=False)

    if st.session_state.conversation_chain is None:
        with st.spinner("Cargando a Gerard... Por favor, espera."):
            st.session_state.conversation_chain = get_conversational_chain()
        if st.session_state.conversation_chain is None:
            st.stop()

    if not st.session_state.user_name:
        # Usar markdown para mostrar el texto en rojo
        st.markdown('<p class="centered-bigger-red-label">PARA COMENZAR A PREGUNTAR FAVOR PRIMERO INTRODUCE TU NOMBRE.</p>', unsafe_allow_html=True)
        st.markdown('<div class="nombre-label">NOMBRE</div>', unsafe_allow_html=True)
        user_input = st.text_input("", key="name_input", label_visibility="collapsed")
        if user_input:
            st.session_state.user_name = user_input
            st.session_state.user_gender = detect_gender(user_input)
            st.rerun()
    else:
        # Mostrar mensaje de bienvenida
        gender_suffix = "A" if st.session_state.user_gender == 'female' else "O"
        welcome_text = f"BIENVENID{gender_suffix} {st.session_state.user_name.upper()}"
        st.markdown(f'<div class="welcome-message">{welcome_text}</div>', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        user_question = st.chat_input(f"Hola {st.session_state.user_name}, ¿en qué puedo ayudarte?")

        if user_question:
            # Formatear la pregunta con animaciones
            formatted_question = f'<span class="question-mark-start">❓</span><span class="question-text">{user_question}</span><span class="question-mark-end">❓</span>'
            st.session_state.chat_history.append({"role": "user", "content": formatted_question})
            with st.chat_message("user"):
                st.markdown(formatted_question, unsafe_allow_html=True)

            with st.chat_message("assistant"):
                # Mostrar animación de OVNI personalizada
                gif_base64 = load_gif_as_base64("assets/ovni.gif")
                if gif_base64:
                    placeholder = st.empty()
                    placeholder.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <img src="data:image/gif;base64,{gif_base64}" width="150" height="150">
                        <p style="color: yellow; font-size: 1.5rem; font-weight: bold; margin-top: 10px;">
                            Buscando...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                session_id = None
                try:
                        # Obtener información de la request (User-Agent, etc.)
                        request_info = {}
                        try:
                            # Intentar obtener headers si están disponibles
                            request_info['user_agent'] = st.context.headers.get('User-Agent', '') if hasattr(st, 'context') and hasattr(st.context, 'headers') else ''
                            request_info['url'] = 'http://localhost:8501'
                        except Exception:
                            request_info['user_agent'] = ''
                            request_info['url'] = 'http://localhost:8501'
                        
                        # Iniciar logging de la interacción
                        session_id = st.session_state.logger.start_interaction(
                            user=st.session_state.user_name,
                            question=user_question,
                            request_info=request_info
                        )
                        
                        # Marcar inicio de consulta RAG
                        st.session_state.logger.mark_phase(session_id, "rag_start")
                        
                        # Marcar inicio de consulta LLM
                        st.session_state.logger.mark_phase(session_id, "llm_start")
                        
                        result = st.session_state.conversation_chain.invoke({"question": user_question})
                        
                        # Marcar fin de consulta LLM
                        st.session_state.logger.mark_phase(session_id, "llm_end")
                        
                        answer = result["answer"]
                        sources = result.get("source_documents", [])
                        
                        # Registrar respuesta
                        st.session_state.logger.log_response(session_id, answer, sources)
                        
                        # Marcar inicio de procesamiento
                        st.session_state.logger.mark_phase(session_id, "processing_start")
                        
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
                        
                        # Marcar fin de procesamiento
                        st.session_state.logger.mark_phase(session_id, "processing_end")
                        
                        # Marcar inicio de render
                        st.session_state.logger.mark_phase(session_id, "render_start")
                        
                        st.markdown(final_answer_html, unsafe_allow_html=True)
                        st.session_state.chat_history.append({"role": "assistant", "content": final_answer_html})
                        
                        # Marcar fin de render
                        st.session_state.logger.mark_phase(session_id, "render_end")
                        
                        # Limpiar el placeholder del OVNI
                        if gif_base64:
                            placeholder.empty()
                        
                        # Finalizar logging con éxito
                        if session_id:
                            st.session_state.logger.end_interaction(session_id, status="success")

                        # --- Funcionalidad de Exportar y Compartir ---
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("📥 Exportar")
                            
                            # Botón HTML
                            st.download_button(
                                label="📄 Descargar como HTML",
                                data=final_answer_html.encode('utf-8'),
                                file_name="respuesta_gerard.html",
                                mime="text/html"
                            )
                            
                            # Botón TXT
                            txt_content = export_to_txt(final_answer_html, user_question)
                            st.download_button(
                                label="📝 Descargar como TXT",
                                data=txt_content.encode('utf-8'),
                                file_name="respuesta_gerard.txt",
                                mime="text/plain"
                            )
                            
                            # Botón Markdown
                            md_content = export_to_markdown(final_answer_html, user_question)
                            st.download_button(
                                label="📋 Descargar como Markdown",
                                data=md_content.encode('utf-8'),
                                file_name="respuesta_gerard.md",
                                mime="text/markdown"
                            )
                            
                            # Botón PDF
                            pdf_content = export_to_pdf(final_answer_html, user_question)
                            st.download_button(
                                label="📕 Descargar como PDF",
                                data=pdf_content,
                                file_name="respuesta_gerard.pdf",
                                mime="application/pdf"
                            )
                        
                        with col2:
                            st.subheader("🔗 Compartir")
                            # Para compartir, usamos una versión de texto plano simple
                            plain_text_for_sharing = re.sub('<[^<]+?>', '', final_answer_html)
                            # Limitar longitud del texto para evitar URLs demasiado largas
                            max_length = 4000
                            if len(plain_text_for_sharing) > max_length:
                                plain_text_for_sharing = plain_text_for_sharing[:max_length] + "..."
                            encoded_text = quote(plain_text_for_sharing)
                            st.markdown(f"[📧 Compartir por Email](mailto:?subject=Respuesta%20de%20Gerard&body={encoded_text})")
                            st.markdown(f"[💬 Compartir en WhatsApp](https://api.whatsapp.com/send?text={encoded_text})")
                            st.markdown(f"[✈️ Compartir en Telegram](https://telegram.me/share/url?text={encoded_text})")

                except Exception as e:
                    # Limpiar el placeholder del OVNI en caso de error
                    if gif_base64:
                        placeholder.empty()
                    st.error(f"Ocurrió un error al procesar tu pregunta: {e}")
                    
                    # Finalizar logging con error
                    if session_id:
                        st.session_state.logger.end_interaction(session_id, status="error", error=str(e))

if __name__ == "__main__":
    main()

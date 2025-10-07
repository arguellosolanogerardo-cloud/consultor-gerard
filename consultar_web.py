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

# Para leer markdown
import markdown

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

def extract_plain_text(html_content):
    """Extrae texto plano del contenido HTML para copiar al portapapeles."""
    # Eliminar todas las etiquetas HTML
    plain_text = re.sub('<[^<]+?>', '', html_content)
    # Decodificar entidades HTML comunes
    plain_text = plain_text.replace('&amp;', '&')
    plain_text = plain_text.replace('&lt;', '<')
    plain_text = plain_text.replace('&gt;', '>')
    plain_text = plain_text.replace('&quot;', '"')
    plain_text = plain_text.replace('&#39;', "'")
    # Limpiar espacios múltiples y líneas vacías excesivas
    plain_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', plain_text)
    plain_text = re.sub(r' +', ' ', plain_text)
    return plain_text.strip()

def export_to_pdf(html_content, user_question):
    """Convierte HTML a PDF para exportación."""
    try:
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
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12
        )
        
        normal_style = styles['BodyText']
        normal_style.fontSize = 10
        normal_style.leading = 13
        
        # Título
        elements.append(Paragraph("GERARD - Respuesta", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"<b>Fecha:</b> {timestamp}", normal_style))
        
        # Escapar caracteres especiales en la pregunta
        safe_question = user_question.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        elements.append(Paragraph(f"<b>Pregunta:</b> {safe_question}", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Convertir HTML a texto plano para evitar problemas
        plain_text = re.sub('<[^<]+?>', '', html_content)
        
        # Dividir en párrafos y agregar al PDF
        paragraphs = plain_text.split('\n')
        for para in paragraphs:
            if para.strip():
                # Escapar caracteres especiales
                safe_para = para.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                try:
                    elements.append(Paragraph(safe_para, normal_style))
                    elements.append(Spacer(1, 0.1*inch))
                except Exception as e:
                    # Si un párrafo específico causa error, continuar con el siguiente
                    continue
        
        # Construir PDF
        doc.build(elements)
        
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        # Si hay error en la generación del PDF, retornar None
        st.error(f"Error al generar PDF: {str(e)}")
        return None

def load_guide_content():
    """Carga el contenido de la guía desde el archivo markdown."""
    try:
        with open("GUIA_MODELOS_PREGUNTA_GERARD.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error al cargar la guía: {e}"

def export_guide_to_pdf():
    """Convierte la guía completa a PDF."""
    try:
        guide_content = load_guide_content()
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        elements = []
        styles = getSampleStyleSheet()
        
        # Convertir markdown a texto plano
        plain_text = re.sub(r'#+\s+', '', guide_content)  # Remover headers markdown
        plain_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', plain_text)  # Remover bold
        plain_text = re.sub(r'`([^`]+)`', r'\1', plain_text)  # Remover code
        
        # Dividir en párrafos
        paragraphs = plain_text.split('\n')
        for para in paragraphs:
            if para.strip():
                safe_para = para.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                try:
                    elements.append(Paragraph(safe_para, styles['BodyText']))
                    elements.append(Spacer(1, 0.1*inch))
                except:
                    continue
        
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error al generar PDF de la guía: {e}")
        return None

def show_help_sidebar():
    """Muestra la sidebar con la guía de ayuda y botones de descarga."""
    with st.sidebar:
        st.markdown("# 📚 AYUDA Y GUÍA DE USO")
        st.markdown("---")
        
        # Sección de descarga de la guía completa
        st.markdown("### 📥 Descargar Guía Completa")
        
        guide_content = load_guide_content()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Descargar como Markdown
            st.download_button(
                label="📄 MD",
                data=guide_content.encode('utf-8'),
                file_name="Guia_Gerard.md",
                mime="text/markdown",
                use_container_width=True,
                help="Descargar guía en formato Markdown"
            )
        
        with col2:
            # Descargar como texto plano
            plain_guide = re.sub(r'[#*`]', '', guide_content)
            st.download_button(
                label="📋 TXT",
                data=plain_guide.encode('utf-8'),
                file_name="Guia_Gerard.txt",
                mime="text/plain",
                use_container_width=True,
                help="Descargar guía en formato texto"
            )
        
        with col3:
            # Descargar como PDF
            pdf_guide = export_guide_to_pdf()
            if pdf_guide:
                st.download_button(
                    label="📕 PDF",
                    data=pdf_guide,
                    file_name="Guia_Gerard.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Descargar guía en formato PDF"
                )
        
        st.markdown("---")
        
        # Tips rápidos
        with st.expander("💡 Tips Rápidos"):
            st.markdown("""
            **Cómo hacer mejores preguntas:**
            - Sé específico con el tema
            - Usa nombres de maestros (ALANISO, AXEL, etc.)
            - Combina términos: "Maestro ALANISO + evacuación"
            - Pregunta por números: "Meditación 107"
            
            **Ejemplos:**
            - ✅ "¿Qué dice sobre la cura milagrosa?"
            - ✅ "Busca mensajes del Maestro AZEN"
            - ❌ "Dime algo" (muy vago)
            """)
        
        # Categorías de búsqueda
        with st.expander("🎯 Categorías de Búsqueda"):
            st.markdown("""
            1. **Por tema:** evacuación, naves, sanación
            2. **Por maestro:** ALANISO, AXEL, ADIEL, etc.
            3. **Por concepto:** Gran Madre, túneles dimensionales
            4. **Por número:** Meditación 107, Mensaje 686
            5. **Por fecha:** Navidad, Reyes Magos
            6. **Comparativas:** Relacionar conceptos
            """)
        
        # Ejemplos de preguntas
        with st.expander("⚡ Ejemplos de Preguntas"):
            st.markdown("""
            **Preguntas efectivas:**
            
            📌 "¿Qué enseñanzas hay sobre la evacuación?"
            
            📌 "Busca mensajes del Maestro ALANISO sobre sanación"
            
            📌 "¿Qué dice la Meditación 555?"
            
            📌 "Explícame sobre las esferas de luz"
            
            📌 "¿Qué conexión hay entre las pirámides y los ángeles?"
            """)
        
        # Maestros disponibles
        with st.expander("👥 Maestros Disponibles"):
            st.markdown("""
            **9 Maestros:**
            - **ALANISO** - Maestro principal
            - **AXEL** - Organizador de naves
            - **ADIEL** - Enfoque en niños
            - **AZOES** - Mensajes específicos
            - **AVIATAR** - Vidas pasadas
            - **ALADIM** - Comunicación
            - **ALIESTRO** - Protección
            - **ALAN** - Sanación
            - **AZEN** - Ejército de luz
            
            **Entidades Superiores:**
            - EL PADRE AMOR
            - GRAN MAESTRO JESÚS
            - LA GRAN MADRE
            """)
        
        # Vocabulario clave
        with st.expander("📖 Vocabulario Clave"):
            st.markdown("""
            **Términos frecuentes:**
            - Evacuación
            - Naves / Esferas de luz
            - Ejército de luz
            - Túnel dimensional
            - Sanación / Cura milagrosa
            - Hermanos cósmicos
            - Cambio de eras
            - Nave nodriza
            - Tercera dimensión
            - Mundos evolucionados
            - Pirámides
            - Mensajes ocultos
            """)
        
        # Consejos avanzados
        with st.expander("🚀 Tips Avanzados"):
            st.markdown("""
            **Aprovecha el modelo Gemini:**
            
            🔹 **Temperatura 0.3** = Respuestas consistentes y precisas
            
            🔹 **Búsqueda iterativa:** Haz preguntas de seguimiento
            
            🔹 **Contexto conversacional:** GERARD recuerda la conversación
            
            🔹 **Comparaciones:** "Compara enseñanzas de ALANISO vs AXEL"
            
            🔹 **Profundización:** "De esa información, profundiza en..."
            """)
        
        # Formato de referencias
        with st.expander("📍 Formato de Referencias"):
            st.markdown("""
            **Cómo leer las citas:**
            
            Las referencias aparecen como:
            ```
            (Nombre archivo - MM:SS)
            ```
            
            **Ejemplo:**
            ```
            (MEDITACION 107 LA CURA MILAGROSA 
             MAESTRO ALANISO - 00:46)
            ```
            
            **Colores:**
            - 🔵 **Azul:** Citas textuales
            - 🟣 **Violeta:** Referencias (archivo + tiempo)
            """)
        
        st.markdown("---")
        st.markdown("**GERARD 3.0** - Analista Investigativo")
        st.markdown("*Modelo: gemini-pro-latest*")

def main():
    st.set_page_config(page_title="GERARD", page_icon="🔮")
    
    # Mostrar sidebar con ayuda
    show_help_sidebar()
    
    st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Orbitron&display=swap');

@keyframes pulse {
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
  100% {
    transform: scale(1);
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

/* Responsive para tablets */
@media (max-width: 768px) {
  .title-gerard {
    font-size: 4rem;
    letter-spacing: 0.3rem;
  }
}

/* Responsive para móviles */
@media (max-width: 480px) {
  .title-gerard {
    font-size: 2.5rem;
    letter-spacing: 0.2rem;
    margin-bottom: 10px;
  }
}

.welcome-message {
  font-family: 'Orbitron', sans-serif;
  font-size: clamp(1.5rem, 5vw, 2.5rem);
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

/* Estilos para párrafos más anchos y responsive */
p {
  max-width: 100%;
  word-wrap: break-word;
  overflow-wrap: break-word;
  margin: 0 auto;
}

ul, ol {
  max-width: 100%;
  word-wrap: break-word;
  overflow-wrap: break-word;
  padding-left: 20px;
}

li {
  word-wrap: break-word;
  overflow-wrap: break-word;
  margin-bottom: 10px;
}

/* Contenedor principal más ancho */
.stChatMessage {
  max-width: 95% !important;
  width: 95% !important;
}

/* Mensajes del chat */
div[data-testid="stChatMessageContent"] {
  max-width: 100% !important;
  width: 100% !important;
}

/* Responsive para diferentes dispositivos */
@media (min-width: 1200px) {
  /* Desktop grande */
  p, ul, ol {
    max-width: 100%;
  }
}

@media (max-width: 1199px) and (min-width: 768px) {
  /* Tablets */
  p, ul, ol {
    max-width: 100%;
    padding: 0 10px;
  }
}

@media (max-width: 767px) {
  /* Móviles */
  p, ul, ol {
    max-width: 100%;
    padding: 0 5px;
    font-size: 0.95rem;
  }
  
  li {
    font-size: 0.9rem;
    margin-bottom: 8px;
  }
}
</style>

<div class="title-gerard">GERARD</div>
''', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align: center; color: #0066CC;">
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
            # Cargar el GIF de pregunta
            pregunta_gif_base64 = load_gif_as_base64("assets/pregunta.gif")
            
            # Formatear la pregunta con el GIF de pregunta al inicio y símbolo animado al final
            if pregunta_gif_base64:
                formatted_question = f'<img src="data:image/gif;base64,{pregunta_gif_base64}" width="30" height="30" style="display: inline-block; margin-right: 10px; animation: rotate 1.5s infinite;"><span class="question-text">{user_question}</span><span class="question-mark-end">❓</span>'
            else:
                # Fallback a emojis si el GIF no se puede cargar
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
                    <div style=\"display: flex; align-items: center; justify-content: center; margin: 20px 0;\">
                        <p style=\"color: yellow; font-size: 1.5rem; font-weight: bold; margin-right: 10px;\">
                            Buscando...
                        </p>
                        <img src=\"data:image/gif;base64,{gif_base64}\" width=\"150\" height=\"150\">
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

                        # --- POST-PROCESAMIENTO ROBUSTO DE REFERENCIAS ---
                        # Crear un mapa completo de todas las fuentes con sus timestamps
                        source_info_map = {}
                        
                        if sources:
                            for doc in sources:
                                source_path = doc.metadata.get("source", "Fuente desconocida")
                                source_file = os.path.basename(source_path)
                                cleaned_name = clean_source_name(source_file)
                                
                                # Extraer timestamp del contenido
                                timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2}),\d{3}', doc.page_content)
                                if timestamp_match:
                                    # Convertir timestamp a formato MM:SS
                                    full_time = timestamp_match.group(1)  # HH:MM:SS
                                    time_parts = full_time.split(':')
                                    # Tomar solo MM:SS
                                    short_time = f"{time_parts[1]}:{time_parts[2]}"
                                    
                                    # Crear la referencia completa
                                    full_reference = f"{cleaned_name} - {short_time}"
                                    
                                    # Guardar múltiples variantes para poder encontrar la referencia
                                    # Extraer número de meditación/mensaje
                                    num_match = re.search(r'(?:MEDITACION|MENSAJE)\s*(\d+)', source_file, re.IGNORECASE)
                                    if num_match:
                                        number = num_match.group(1)
                                        # Guardar por número
                                        source_info_map[number] = full_reference
                                        # Guardar también por nombre de archivo parcial
                                        source_info_map[cleaned_name] = full_reference
                                        # Guardar por timestamp
                                        source_info_map[short_time] = full_reference
                                        source_info_map[full_time] = full_reference
                        
                        # Paso 1: Reemplazar todas las referencias encontradas en el texto
                        modified_answer = answer
                        
                        # Buscar y reemplazar patrones comunes de referencias:
                        # Patrón 1: (263.srt - 27:09) o similar
                        modified_answer = re.sub(
                            r'\((\d+)\.srt\s*-\s*(\d{2}:\d{2})\)',
                            lambda m: f"({source_info_map.get(m.group(1), source_info_map.get(m.group(2), f'{m.group(1)}.srt - {m.group(2)}'))})",
                            modified_answer
                        )
                        
                        # Patrón 2: Solo números entre paréntesis (107), (263), etc.
                        modified_answer = re.sub(
                            r'\((\d+)\)',
                            lambda m: f"({source_info_map.get(m.group(1), m.group(1))})",
                            modified_answer
                        )
                        
                        # Patrón 3: Timestamps completos (HH:MM:SS)
                        modified_answer = re.sub(
                            r'\((\d{2}:\d{2}:\d{2})\)',
                            lambda m: f"({source_info_map.get(m.group(1), m.group(1))})",
                            modified_answer
                        )
                        
                        # Patrón 4: Timestamps cortos (MM:SS)
                        modified_answer = re.sub(
                            r'\((\d{2}:\d{2})\)',
                            lambda m: f"({source_info_map.get(m.group(1), m.group(1))})",
                            modified_answer
                        )
                        
                        # Paso 2: Aplicar color violeta a TODO el contenido entre paréntesis
                        formatted_answer = re.sub(
                            r'\(([^)]+)\)',
                            lambda m: f"(<span style='color: violet;'>{m.group(1)}</span>)",
                            modified_answer
                        )
                        
                        final_answer_html = f"<p>{formatted_answer}</p>"
                        
                        # Agregar sección de citas textuales
                        if sources:
                            final_answer_html += "<h3>--- Citas Textuales ---</h3>"
                            final_answer_html += "<ul>"
                            
                            for doc in sources:
                                source_path = doc.metadata.get("source", "Fuente desconocida")
                                source_file = os.path.basename(source_path)
                                cleaned_name = clean_source_name(source_file)
                                
                                # Extraer timestamp
                                timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})', doc.page_content)
                                if timestamp_match:
                                    timestamp = timestamp_match.group(0)
                                    short_timestamp = re.sub(r',\d{3}', '', timestamp)
                                    source_info = f"(Fuente: {cleaned_name}, Timestamp: {short_timestamp})"
                                else:
                                    source_info = f"(Fuente: {cleaned_name}, Timestamp: No disponible)"
                                
                                # Limpiar el contenido del SRT
                                quote_text = clean_srt_content(doc.page_content)
                                
                                # Formatear con colores
                                highlighted_quote = f'<span style="color: #0066CC;">{quote_text}</span>'
                                violet_source = f' <span style="color: violet;">{source_info}</span>'
                                
                                final_answer_html += f"<li>{highlighted_quote}{violet_source}</li>"
                            
                            final_answer_html += "</ul>"
                        # --- FIN DE LA MODIFICACIÓN ---
                        
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

                except Exception as e:
                    # Limpiar el placeholder del OVNI en caso de error
                    if gif_base64:
                        placeholder.empty()
                    st.error(f"Ocurrió un error al procesar tu pregunta: {e}")
                    
                    # Finalizar logging con error
                    if session_id:
                        st.session_state.logger.end_interaction(session_id, status="error", error=str(e))
            
            # --- Funcionalidad de Exportar (fuera del contexto del mensaje) ---
            # Solo mostrar si hay historial de chat con respuestas
            if len(st.session_state.chat_history) > 0:
                # Obtener la última respuesta del asistente
                last_assistant_message = None
                for msg in reversed(st.session_state.chat_history):
                    if msg["role"] == "assistant":
                        last_assistant_message = msg["content"]
                        break
                
                if last_assistant_message:
                    st.markdown("---")
                    st.subheader("📥 Exportar última respuesta")
                    
                    # Crear tres columnas para los botones
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Botón HTML
                        st.download_button(
                            label="📄 Descargar HTML",
                            data=last_assistant_message.encode('utf-8'),
                            file_name="respuesta_gerard.html",
                            mime="text/html",
                            use_container_width=True,
                            key=f"html_btn_{len(st.session_state.chat_history)}"
                        )
                    
                    with col2:
                        # Botón PDF
                        pdf_content = export_to_pdf(last_assistant_message, user_question)
                        if pdf_content:
                            st.download_button(
                                label="📕 Descargar PDF",
                                data=pdf_content,
                                file_name="respuesta_gerard.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"pdf_btn_{len(st.session_state.chat_history)}"
                            )
                    
                    with col3:
                        # Botón para copiar texto plano
                        plain_text = extract_plain_text(last_assistant_message)
                        # Agregar pregunta al inicio del texto
                        full_text = f"Pregunta: {user_question}\n\n{plain_text}"
                        
                        # Usar download_button como alternativa para copiar
                        st.download_button(
                            label="📋 Copiar Texto",
                            data=full_text.encode('utf-8'),
                            file_name="respuesta_gerard.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key=f"txt_btn_{len(st.session_state.chat_history)}"
                        )
                    
                    # Agregar área de texto expandible para copiar manualmente
                    with st.expander("📝 Ver texto completo para copiar"):
                        plain_text_for_copy = extract_plain_text(last_assistant_message)
                        st.text_area(
                            "Selecciona y copia el texto:",
                            value=f"Pregunta: {user_question}\n\n{plain_text_for_copy}",
                            height=300,
                            key=f"text_area_{len(st.session_state.chat_history)}"
                        )
                        st.info("💡 Consejo: Haz clic dentro del cuadro de texto, presiona Ctrl+A (o Cmd+A en Mac) para seleccionar todo, luego Ctrl+C (o Cmd+C) para copiar.")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import os
import json
import re
import colorama
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from typing import Any, Iterable, List, Pattern
import streamlit as st
import requests  # Para obtener la IP y geolocalización
import io
import textwrap

# Intentar importar reportlab para generar PDFs; si no está disponible, lo detectamos y mostramos instrucciones
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        REPORTLAB_PLATYPUS = True
    except Exception:
        REPORTLAB_PLATYPUS = False

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# --- Configuración Inicial ---
colorama.init(autoreset=True)
load_dotenv()

# --- Carga de Modelos y Base de Datos (con caché de Streamlit) ---
@st.cache_resource
def load_resources():
    # Preferir la variable de entorno; en Streamlit tomar como fallback st.secrets
    api_key = os.environ.get("GOOGLE_API_KEY")
    try:
        if not api_key and hasattr(st, "secrets"):
            api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        # En entornos sin Streamlit secrets esto puede fallar; ignoramos
        pass
    if not api_key:
        st.error(
            "Error: La variable de entorno GOOGLE_API_KEY no está configurada. Añade la clave a las variables de entorno o a Streamlit Secrets."
        )
        st.stop()

    # Pasar la API key explícitamente evita que la librería intente usar ADC
    with st.spinner('Inicializando LLM y embeddings...'):
        llm = GoogleGenerativeAI(model="models/gemini-2.5-pro", google_api_key=api_key)

    # Cargar índice FAISS persistido en 'faiss_index' con spinner
    try:
        with st.spinner('Cargando índice FAISS (esto puede tardar varios segundos)...'):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            faiss_vs = FAISS.load_local(folder_path="faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"No fue posible cargar el índice FAISS: {e}")
        st.stop()

    return llm, faiss_vs

llm, vectorstore = load_resources()

# --- Lógica de GERARD (sin cambios) ---
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
  {{ "type": "normal", "content": ". (Fuente: archivo.srt, Timestamp: 00:01:23 --> 00:01:25)" }}
]
--- REGLA DE CITA ---
Incluye las citas de la fuente DENTRO del "content" de un objeto de tipo "normal". El formato es: `(Fuente: nombre_del_archivo.srt, Timestamp: HH:MM:SS --> HH:MM:SS)`.
Comienza tu labor, GERARD. Responde únicamente con el array JSON.
--- FIN DE INSTRUCCIONES DE PERSONALIDAD ---
Basándote ESTRICTAMENTE en las reglas y el contexto de abajo, responde la pregunta del usuario.
<contexto>
{context}
</contexto>
Pregunta del usuario: {input}
""")

def get_cleaning_pattern() -> Pattern:
    texts_to_remove = [
        '[Spanish (auto-generated)]', '[DownSub.com]', '[Música]', '[Aplausos]'
    ]
    robust_patterns = [r'\[\s*' + re.escape(text[1:-1]) + r'\s*\]' for text in texts_to_remove]
    return re.compile(r'|'.join(robust_patterns), re.IGNORECASE)

cleaning_pattern = get_cleaning_pattern()

def format_docs_with_metadata(docs: Iterable[Any]) -> str:
    """Formatea una secuencia de documentos recuperados y limpia su contenido.

    docs: iterable de objetos con atributos `metadata` (dict) y `page_content` (str).
    Devuelve una única cadena con todos los documentos formateados.
    """
    formatted_strings: List[str] = []
    for doc in docs:
        source_filename = os.path.basename(doc.metadata.get('source', 'Desconocido'))
        texts_to_remove_from_filename = ["[Spanish (auto-generated)]", "[DownSub.com]"]
        for text_to_remove in texts_to_remove_from_filename:
            source_filename = source_filename.replace(text_to_remove, "")
        source_filename = re.sub(r'\s+', ' ', source_filename).strip()
        cleaned_content = cleaning_pattern.sub('', doc.page_content)
        cleaned_content = re.sub(r'(\d{2}:\d{2}:\d{2}),\d{3}', r'\1', cleaned_content)
        cleaned_content = "\n".join(line for line in cleaned_content.split('\n') if line.strip())
        if cleaned_content:
            formatted_strings.append(f"Fuente: {source_filename}\nContenido:\n{cleaned_content}")
    return "\n\n---\n\n".join(formatted_strings)

llm, vectorstore = load_resources()
retriever = vectorstore.as_retriever()
retrieval_chain = (
    {"context": retriever | format_docs_with_metadata, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Funciones de Geolocalización y Registro ---
@st.cache_data
def get_user_location() -> str:
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        data = response.json()
        ip = data.get('ip', 'No disponible')
        city = data.get('city', 'Desconocida')
        country = data.get('country', 'Desconocido')
        return f"{city}, {country} (IP: {ip})"
    except Exception:
        return "Ubicación no disponible"

def get_clean_text_from_json(json_string: str) -> str:
    try:
        match = re.search(r'\[.*\]', json_string, re.DOTALL)
        if not match:
            return json_string

        data = json.loads(match.group(0))
        return "".join([item.get("content", "") for item in data])
    except Exception:
        return json_string


def detect_gender_from_name(name: str) -> str:
    """Heurística simple para detectar género a partir del primer nombre.
    Regla principal: termina en 'a' -> Femenino, termina en 'o' -> Masculino.
    Usa listas de excepciones comunes para mejorar la precisión.
    Devuelve: 'Masculino', 'Femenino' o 'No especificar'.
    """
    if not name or not name.strip():
        return 'No especificar'
    # Normalizar y tomar primer token
    first = name.strip().split()[0].lower()
    # Quitar caracteres no alfabéticos (mantener acentos y ñ)
    first = re.sub(r"[^a-záéíóúüñ]", "", first)

    # Listas de nombres comunes (no exhaustivas)
    male_names = {"juan","carlos","pedro","jose","luis","miguel","axel","alan","adriel","adiel","alaniso","aladio","adolfo"}
    female_names = {"maria","ana","laura","mariana","isabela","isabella","sofia","sofia"}

    if first in male_names:
        return 'Masculino'
    if first in female_names:
        return 'Femenino'

    # Regla por terminación (heurística fuerte en español)
    if first.endswith(('a','á')):
        return 'Femenino'
    if first.endswith(('o','ó')):
        return 'Masculino'

    # Nombres neutros o no detectables
    return 'No especificar'

def save_to_log(user: str, question: str, answer_json: str, location: str) -> None:
    clean_answer = get_clean_text_from_json(answer_json)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("gerard_log.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Conversación del {timestamp} ---\n")
        f.write(f"Usuario: {user}\n")
        f.write(f"Ubicación: {location}\n")
        f.write(f"Pregunta: {question}\n")
        f.write(f"Respuesta de GERARD: {clean_answer}\n")
        f.write("="*40 + "\n\n")

def get_conversation_text() -> str:
    conversation = []
    for message in st.session_state.get('messages', []):
        content_html = message["content"]
        # Extraer texto plano de la forma más simple posible
        text_content = re.sub(r'<[^>]+>', '', content_html).strip()
        
        if message["role"] == "user":
            # Para el usuario, el texto relevante está en el span uppercase
            match = re.search(r'<span style="text-transform: uppercase;.*?">(.*?)</span>', content_html)
            if match:
                text_content = match.group(1).strip()
            conversation.append(f"Usuario: {text_content}")
        else:
            # Para el asistente, quitar el nombre de usuario que se añade al principio
            user_name_placeholder = f"{st.session_state.get('user_name', '')}:"
            if text_content.startswith(user_name_placeholder):
                 text_content = text_content[len(user_name_placeholder):].strip()
            conversation.append(f"GERARD: {text_content}")
            
    return "\n\n".join(conversation)

def generate_download_filename() -> str:
    user_questions = []
    for message in st.session_state.get('messages', []):
        if message["role"] == "user":
            match = re.search(r'<span style="text-transform: uppercase;.*?">(.*?)</span>', message['content'])
            if match:
                user_questions.append(match.group(1).strip())

    if not user_questions:
        base_name = "conversacion"
    else:
        # Unir preguntas con un guion bajo para más claridad
        base_name = "_".join(user_questions)

    # Sanitizar y truncar
    sanitized_name = re.sub(r'[\\/:*?"<>|]', '', base_name)
    truncated_name = sanitized_name[:200].strip()

    user_name = st.session_state.get('user_name', 'usuario').upper()
    
    # Formato final: PREGUNTAS_USUARIO.txt
    return f"{truncated_name}_{user_name}.txt"


def _escape_ampersand(text: str) -> str:
    return text.replace('&', '&amp;')


def _convert_spans_to_font_tags(html: str) -> str:
    """Reemplaza <span style="color:...">texto</span> por <font color="...">texto</font> para que reportlab Paragraph lo soporte.

    No soportamos estilos complejos; se intenta preservar el color de fuente.
    """
    # Normalizar algunos cierres y saltos
    s = html
    # Reemplazar span color (hex o nombre)
    s = re.sub(r'<span\s+style="[^"]*color\s*:\s*([^;\"]+)[^\"]*">(.*?)</span>', lambda m: f"<font color=\"{m.group(1).strip()}\">{m.group(2)}</font>", s, flags=re.DOTALL)
    # Reemplazar any remaining <span> without color -> remove span
    s = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', s, flags=re.DOTALL)
    # Asegurar que los saltos de línea HTML sean <br/> para Paragraph
    s = s.replace('\n', '<br/>')
    s = s.replace('<br>', '<br/>')
    # Evitar caracteres & que rompan XML interno
    s = _escape_ampersand(s)
    return s


def _format_header(title_base: str, user_name: str | None, max_len: int = 220):
    """Construye un encabezado que contiene el título, el nombre en negrita y la fecha, limitado a max_len caracteres.

    Devuelve una tupla (header_html, header_plain).
    """
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    user_name = (user_name or 'usuario').strip()
    plain = f"{title_base} - {user_name} {date_str}"
    if len(plain) > max_len:
        plain = plain[: max_len - 3].rstrip() + '...'
    # Para HTML, ponemos el nombre en negrita
    # Intentar reemplazar first occurrence of user_name in plain with bold; si truncado puede no contener name
    if user_name and user_name in plain:
        html = plain.replace(user_name, f"<b>{user_name}</b>", 1)
    else:
        html = plain
    return html, plain


def generate_pdf_from_html(html_content: str, title_base: str = "Conversacion GERARD", user_name: str | None = None) -> bytes:
    """Genera un PDF en memoria a partir de HTML simple (etiquetas básicas) preservando colores de fuente.

    Usa reportlab Platypus Paragraph con tags <font color="...">.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("La librería 'reportlab' no está instalada. Instálala con: pip install reportlab")
    if not REPORTLAB_PLATYPUS:
        # Si platypus no está disponible, caer al generador de texto plano
        return generate_pdf_bytes_text(_strip_html_tags(html_content), title_base=title_base, user_name=user_name)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20, leftMargin=20, topMargin=30, bottomMargin=20)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    normal.fontName = 'Helvetica'
    normal.fontSize = 10
    normal.leading = 12

    story = []
    # Header (título + nombre en negrita + fecha) limitado a 220 chars
    header_html, header_plain = _format_header(title_base, user_name, max_len=220)
    title_style = styles.get('Heading2', normal)
    story.append(Paragraph(header_html, title_style))
    story.append(Spacer(1, 6))

    body = _convert_spans_to_font_tags(html_content)

    # Paragraph acepta un fragmento con tags limitados (<b>, <i>, <u>, <font color="...">, <br/>)
    try:
        story.append(Paragraph(body, normal))
    except Exception:
        # Fallback: limpiar HTML y usar texto simple
        plain = re.sub(r'<[^>]+>', '', html_content)
        story.append(Paragraph(plain.replace('&', '&amp;'), normal))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_pdf_bytes_text(text: str, title_base: str = "Conversacion GERARD", user_name: str | None = None) -> bytes:
    """Fallback simple: genera PDF plano a partir de texto sin formato (mantener función previa)."""
    buffer = io.BytesIO()
    page_width, page_height = A4
    c = canvas.Canvas(buffer, pagesize=A4)
    left_margin = 40
    right_margin = 40
    top_margin = 40
    bottom_margin = 40
    # Header: título + nombre en negrita + fecha (limitado a 220 chars)
    header_html, header_plain = _format_header(title_base, user_name, max_len=220)
    # Dibujar parte inicial (título y nombre en negrita separado por un guion)
    # Si header_plain contiene el user_name, dibujamos antes del nombre en normal y luego el nombre en negrita
    if user_name and user_name in header_plain:
        prefix, _, suffix = header_plain.partition(user_name)
        c.setFont("Helvetica-Bold", 14)
        # Dibujar prefijo + (usaremos fuente normal para prefijo) -> ajustar: dibujar prefix en normal
        c.setFont("Helvetica", 12)
        c.drawString(left_margin, page_height - top_margin, prefix.strip())
        # dibujar nombre en negrita seguido de fecha/suffix
        x = left_margin + stringWidth(prefix.strip() + ' ', "Helvetica", 12)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, page_height - top_margin, user_name)
        x += stringWidth(user_name + ' ', "Helvetica-Bold", 12)
        c.setFont("Helvetica", 12)
        c.drawString(x, page_height - top_margin, suffix.strip())
    else:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, page_height - top_margin, header_plain)
    c.setFont("Helvetica", 10)
    max_width = page_width - left_margin - right_margin
    from reportlab.pdfbase.pdfmetrics import stringWidth
    y = page_height - top_margin - 20
    line_height = 12
    for paragraph in text.split('\n'):
        if not paragraph:
            y -= line_height
            if y < bottom_margin:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = page_height - top_margin
            continue
        words = paragraph.split(' ')
        line = ''
        for w in words:
            candidate = (line + ' ' + w).strip() if line else w
            if stringWidth(candidate, "Helvetica", 10) <= max_width:
                line = candidate
            else:
                c.drawString(left_margin, y, line)
                y -= line_height
                if y < bottom_margin:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = page_height - top_margin
                line = w
        if line:
            c.drawString(left_margin, y, line)
            y -= line_height
            if y < bottom_margin:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = page_height - top_margin
    c.save()
    buffer.seek(0)
    return buffer.read()


def _strip_html_tags(html: str) -> str:
    return re.sub(r'<[^>]+>', '', html)


# --- Interfaz de Usuario con Streamlit ---
st.set_page_config(page_title="GERARD", layout="centered")

# --- Avatares personalizados ---
user_avatar = "https://api.iconify.design/line-md/question-circle.svg?color=%2358ACFA"
assistant_avatar = "https://api.iconify.design/mdi/ufo-outline.svg?color=%238A2BE2"


# --- Estilos CSS y Título ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;800&display=swap');
.title-style {
    /* Tipografía moderna y mayor tamaño */
    font-family: 'Poppins', 'Orbitron', sans-serif;
    font-size: 8em; /* un poco más grande */
    text-align: center;
    color: #8A2BE2; /* Violeta */
    padding-bottom: 20px;
    line-height: 0.9;
    /* pulso suave (palpitante) */
    animation: pulse-title 2s infinite ease-in-out;
    text-shadow: 0 6px 18px rgba(138,43,226,0.15);
}
.welcome-text {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5em;
    text-align: center;
    color: #28a745; /* Green */
    padding-bottom: 5px;
}
.sub-welcome-text {
    text-align: center;
    font-size: 1.1em;
    margin-top: -15px;
    padding-bottom: 20px;
}
.intro-text {
    text-transform: uppercase;
    text-align: center;
    color: #58ACFA; /* Azul claro */
    font-size: 2.6em;
    padding-bottom: 20px;
}
.loader-container {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding-top: 5px;
}
.dot {
    height: 10px;
    width: 10px;
    margin: 0 3px;
    background-color: #8A2BE2; /* Violeta */
    border-radius: 50%;
    display: inline-block;
    animation: bounce 1.4s infinite ease-in-out both;
}
.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1.0); }
}

@keyframes pulse-title {
    0% { transform: scale(1); }
    50% { transform: scale(1.06); }
    100% { transform: scale(1); }
}

/* --- ¡NUEVA ANIMACIÓN CSS! --- */
.pulsing-q {
    font-size: 1.5em; /* 24px */
    color: red;
    font-weight: bold;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.25); opacity: 0.75; }
    100% { transform: scale(1); opacity: 1; }
}
/* Clase reutilizable para texto verde pulsante */
.green-pulse {
    color: #28a745; /* verde */
    font-weight: bold;
    font-size: 2em;
    animation: pulse 1.2s infinite;
}

/* Media queries para móviles (mejor legibilidad en Android/iOS antiguos y modernos) */
@media (max-width: 1200px) {
    .title-style { font-size: 5.2em; }
    .intro-text { font-size: 1.6em; }
}
@media (max-width: 800px) {
    .title-style { font-size: 3.2em; }
    .intro-text { font-size: 1.15em; }
    .welcome-text { font-size: 1.6em; }
}
@media (max-width: 480px) {
    .title-style { font-size: 2.4em; }
    .intro-text { font-size: 1.0em; }
    .welcome-text { font-size: 1.2em; }
    .green-pulse { font-size: 1.2em; }
}
@media (max-width: 360px) {
    .title-style { font-size: 2.0em; }
    .intro-text { font-size: 0.95em; }
    .green-pulse { font-size: 1.0em; }
}
</style>
<div class="title-style">GERARD</div>
""", unsafe_allow_html=True)

# (UI refinements removed; restored original behavior)

location = get_user_location()

if 'user_name' not in st.session_state:
    st.session_state.user_name = ''
if 'user_gender' not in st.session_state:
    st.session_state.user_gender = 'No especificar'
if 'messages' not in st.session_state:
    st.session_state.messages = []

if not st.session_state.user_name:
    st.markdown("""
    <p class="intro-text" style="font-size:1.8em; line-height:1.05;">
    Hola, soy GERARD tu asistente especializado en los mensajes y meditaciones de los 9 Maestros: <strong>ALANISO, AXEL, ALAN, AZEN, AVIATAR, ALADIM, ADIEL, AZOES Y ALIESTRO</strong>.
    <br>
    Las tres grandes energias el Padre Amor, la Gran Madre y el Gran Maestro Jesus.
    </p>
    <p style="text-align:center; margin-top:12px; font-size:1.25em; text-transform:uppercase; font-weight:bold;">
    TE AYUDARE A ENCONTRAR CON PRECISIÓN EL MINUTO Y SEGUNDO EXACTO EN CADA AUDIO O ENSEÑANZAS QUE YA HAYAS ESCUCHADO ANTERIORMENTE ENTRE LAS CIENTOS DE MEDITACIONES Y MENSAJES CANALIZADOS POR SARITA OTERO. PERO QUE EN EL MOMENTO NO RECUERDAS EXACTAMENTE Y ESTÉS BUSCANDO EN TU ARDUO ESTUDIO DE ENCONTRAR NUEVOS MENSAJES DENTRO DE LOS MENSAJES COMO ESTUDIANTE ALANISO AVANZADO.
    <br>
    <span style="color:red;">PARA COMENZAR, POR FAVOR INTRODUCE TU NOMBRE Y TE DARÉ ACCESO PARA QUE PUEDAS HACER TUS PREGUNTAS.</span>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="text-align:center; margin-top:8px;"><span class="green-pulse">TU NOMBRE</span></div>', unsafe_allow_html=True)
    user_name_input = st.text_input("Tu Nombre", key="name_inputter", label_visibility="collapsed")
    if user_name_input:
        st.session_state.user_name = user_name_input.upper()
        # Detección automática del género desde el nombre
        detected = detect_gender_from_name(user_name_input)
        # Permitir confirmar o corregir la detección
        confirm_options = [detected]
        for opt in ("Masculino", "Femenino", "No especificar"):
            if opt not in confirm_options:
                confirm_options.append(opt)

        chosen = st.selectbox("Detecté el género: seleccione para confirmar o corregir", options=confirm_options, index=0, key="gender_confirm")
        st.session_state.user_gender = chosen
        st.rerun()
else:
    # Construir bienvenida flexible según género seleccionado
    gender = st.session_state.get('user_gender', 'No especificar')
    if gender == 'Masculino':
        bienvenida = 'BIENVENIDO'
    elif gender == 'Femenino':
        bienvenida = 'BIENVENIDA'
    else:
        bienvenida = 'BIENVENID@'

    st.markdown(f"""
    <div class="welcome-text">{bienvenida} {st.session_state.user_name}</div>
    <p class="sub-welcome-text">AHORA YA PUEDES REALIZAR TUS PREGUNTAS EN LA PARTE INFERIOR</p>
    """, unsafe_allow_html=True)

# --- Botón de prueba: generar un PDF de ejemplo y ofrecer descarga inmediata ---
if REPORTLAB_AVAILABLE:
    try:
        if st.button("Generar PDF de prueba (demo)"):
            demo_user = st.session_state.get('user_name') or 'PRUEBA_USER'
            demo_html = f'<strong style="color:#28a745;">{demo_user}:</strong> Este es un PDF de prueba con texto normal y una parte <span style="color:yellow; background-color:#333;">ENFATIZADA</span>. (Fuente: ejemplo.srt, Timestamp: 00:00:10 --> 00:00:12)'
            demo_html += f"<br/><br/><span style=\"color:#28a745;\">Usuario: {demo_user}</span>"
            try:
                pdf_bytes = generate_pdf_from_html(demo_html, title_base=f"PDF demo - {demo_user}", user_name=demo_user)
                demo_name = f"sample_QA_{demo_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(label="📄 Descargar PDF demo", data=pdf_bytes, file_name=demo_name, mime="application/pdf")
            except Exception as e:
                st.error(f"No se pudo generar el PDF de prueba: {e}")
    except Exception:
        # No queremos que la UI se rompa por este botón en entornos extraños
        pass

# --- Mostrar historial con avatares personalizados ---
for message in st.session_state.messages:
    avatar = user_avatar if message["role"] == "user" else assistant_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- Botón de Descarga ---
if st.session_state.messages:
    conversation_text = get_conversation_text()
    file_name = generate_download_filename()

    col1, col2 = st.columns([1,1])
    with col1:
        st.download_button(
            label="⬇️ Descargar Conversación (TXT)",
            data=conversation_text,
            file_name=file_name,
            mime="text/plain",
            key="download_button"
        )
    with col2:
        pdf_filename = file_name.rsplit('.',1)[0] + '.pdf'
        if REPORTLAB_AVAILABLE:
            try:
                # Para la exportación completa: construiremos el texto concatenando preguntas y respuestas
                # y aplicaremos wrapping a 200 caracteres para que no se corten palabras arbitrariamente.
                convo_lines = []
                for msg in st.session_state.messages:
                    role = msg.get('role')
                    content_html = msg.get('content', '')
                    plain = re.sub(r'<[^>]+>', '', content_html).strip()
                    if role == 'user':
                        wrapped = textwrap.fill(plain, width=200)
                        convo_lines.append(f"Pregunta: {wrapped}")
                    else:
                        wrapped = textwrap.fill(plain, width=200)
                        convo_lines.append(f"Respuesta: {wrapped}")

                # Añadir al final el nombre de la persona que realizó las preguntas
                user_name_for_file = st.session_state.get('user_name', 'usuario')
                convo_lines.append('')
                convo_lines.append(f"Usuario que realizó las preguntas: {user_name_for_file}")

                conversation_full = '\n\n'.join(convo_lines)

                pdf_filename = f"{user_name_for_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_bytes = generate_pdf_bytes_text(conversation_full, title_base=f"Conversación - {user_name_for_file}")
                st.download_button(
                    label="📄 Exportar PDF",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    key="download_pdf_button"
                )
            except Exception as e:
                st.error(f"No se pudo generar el PDF: {e}")
        else:
            st.info("Exportar a PDF no disponible. Instala reportlab: `pip install reportlab` para habilitar esta función.")

# --- Input del usuario con avatares personalizados ---
if prompt_input := st.chat_input("Escribe tu pregunta aquí..."):
    if not st.session_state.user_name:
        st.warning("Por favor, introduce tu nombre para continuar.")
    else:
        # --- ¡AQUÍ ESTÁ EL CAMBIO! ---
        # Se reemplaza la imagen por un texto animado con CSS
        styled_prompt = f"""
        <div style="display: flex; align-items: center; justify-content: flex-start;">
            <span style="text-transform: uppercase; color: orange; margin-right: 8px; font-weight: bold;">{prompt_input}</span>
            <span class="pulsing-q">?</span>
        </div>
        """
        
        st.session_state.messages.append({"role": "user", "content": styled_prompt})
        
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(styled_prompt, unsafe_allow_html=True)

        with st.chat_message("assistant", avatar=assistant_avatar):
            response_placeholder = st.empty()
            loader_html = """
            <div class="loader-container">
                <span class="dot"></span><span class="dot"></span><span class="dot"></span>
                <span style='margin-left: 10px; font-style: italic; color: #888;'>Buscando...</span>
            </div>
            """
            response_placeholder.markdown(loader_html, unsafe_allow_html=True)

            try:
                answer_json = retrieval_chain.invoke(prompt_input)
                save_to_log(st.session_state.user_name, prompt_input, answer_json, location)
                
                match = re.search(r'\[.*\]', answer_json, re.DOTALL)
                if not match:
                    st.error("La respuesta del modelo no fue un JSON válido.")
                    response_html = f'<p style="color:red;">{answer_json}</p>'
                else:
                    data = json.loads(match.group(0))
                    response_html = f'<strong style="color:#28a745;">{st.session_state.user_name}:</strong> '
                    for item in data:
                        content_type = item.get("type", "normal")
                        content = item.get("content", "")
                        if content_type == "emphasis":
                            response_html += f'<span style="color:yellow; background-color: #333; border-radius: 4px; padding: 2px 4px;">{content}</span>'
                        else:
                            content_html = re.sub(r'(\(.*?\))', r'<span style="color:#87CEFA;">\1</span>', content)
                            response_html += content_html
                
                response_placeholder.markdown(response_html, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response_html})

                # --- Ofrecer descarga del último intercambio (pregunta + respuesta) ---
                try:
                    # Texto plano para el archivo
                    def html_to_text(html: str) -> str:
                        return re.sub(r'<[^>]+>', '', html).strip()

                    user_text = prompt_input.strip()
                    assistant_text = html_to_text(response_html)
                    single_qa_text = f"Pregunta: {user_text}\n\nRespuesta:\n{assistant_text}\n"

                    # Botón de descarga PDF que preserva colores (HTML -> PDF)
                    if REPORTLAB_AVAILABLE:
                        try:
                            # Anexar nombre del usuario al final del HTML para que aparezca en el PDF
                            user_name_for_file = st.session_state.get('user_name','usuario')
                            html_for_pdf = response_html + f"<br/><br/><span style=\"color:#28a745;\">Usuario: {user_name_for_file}</span>"
                            pdf_bytes = generate_pdf_from_html(html_for_pdf, title_base=f"Q&A - {user_name_for_file}", user_name=user_name_for_file)
                            pdf_name = f"QA_{user_name_for_file[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            st.download_button(
                                label="📄 Guardar Q/A (PDF)",
                                data=pdf_bytes,
                                file_name=pdf_name,
                                mime="application/pdf",
                                key=f"download_last_pdf_{len(st.session_state.messages)}"
                            )
                        except Exception as e:
                            st.error(f"Error generando PDF del intercambio: {e}")
                    else:
                        st.caption("Instala reportlab (`pip install reportlab`) para habilitar exportar Q/A en PDF.")
                except Exception:
                    # No queremos que una falla aquí rompa la experiencia principal
                    pass

            except Exception as e:
                response_placeholder.error(f"Ocurrió un error al procesar tu pregunta: {e}")


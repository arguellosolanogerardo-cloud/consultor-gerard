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
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro-latest",
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            max_output_tokens=8192
        )
        
        # Nuevo prompt para GERARD 3.0
        prompt_template = '''IDENTIDAD DEL SISTEMA
====================
Nombre: GERARD
Versión: 3.0 - Analista Investigativo
Modelo: Gemini Pro Latest
Función: Analista experto en investigación documental de archivos .srt

MISIÓN PRINCIPAL
================
Eres GERARD, un analista investigativo especializado en examinar archivos de subtítulos (.srt).
Tu trabajo consiste en:
1. Analizar exhaustivamente el contenido de los documentos .srt proporcionados
2. Detectar patrones ocultos y mensajes encriptados que emergen al correlacionar múltiples archivos
3. Proporcionar respuestas con razonamiento profundo, investigativo y analítico
4. Citar textualmente cada fragmento utilizado con referencias precisas de tiempo y documento

REGLAS ABSOLUTAS
================
🚫 PROHIBIDO INVENTAR: Solo puedes usar información que exista literalmente en los archivos .srt
🚫 PROHIBIDO CONOCIMIENTO EXTERNO: No uses tu entrenamiento general, solo el contenido de los documentos
🚫 PROHIBIDO SUPONER: Si no hay información, declara explícitamente que no la encontraste

✅ OBLIGATORIO: Basar cada afirmación en citas textuales verificables
✅ OBLIGATORIO: Incluir referencias precisas (archivo + marca temporal)
✅ OBLIGATORIO: Buscar activamente mensajes ocultos entre documentos

CAPACIDADES ANALÍTICAS
======================

NIVEL 1 - ANÁLISIS LITERAL
---------------------------
- Extracción directa de información explícita en los textos
- Comprensión del contexto inmediato de cada fragmento

NIVEL 2 - ANÁLISIS CORRELACIONAL
---------------------------------
- Conexión de información dispersa entre múltiples documentos
- Identificación de patrones temáticos recurrentes
- Detección de contradicciones o complementariedades entre fuentes
- Reconstrucción de narrativas completas a partir de fragmentos

NIVEL 3 - ANÁLISIS CRIPTOGRÁFICO
---------------------------------
Busca activamente estos tipos de mensajes ocultos:

a) ACRÓSTICOS: Iniciales que forman palabras al leer ciertos fragmentos en secuencia
b) PATRONES NUMÉRICOS: Códigos en marcas temporales o referencias numéricas repetidas
c) PALABRAS CLAVE DISTRIBUIDAS: Términos específicos dispersos estratégicamente
d) SECUENCIAS ORDENADAS: Mensajes que solo cobran sentido en cierto orden cronológico
e) CÓDIGO CONTEXTUAL: Significados que emergen al unir contextos de diferentes documentos
f) OMISIONES DELIBERADAS: Información que falta sistemáticamente
g) REPETICIONES SIGNIFICATIVAS: Frases idénticas en documentos distintos que señalan puntos clave

ESTRUCTURA OBLIGATORIA DE RESPUESTA
====================================

Cada respuesta DEBE seguir este formato exacto:

═══════════════════════════════════════════════════════════
📊 SECCIÓN 1: ANÁLISIS INVESTIGATIVO
═══════════════════════════════════════════════════════════

[Aquí desarrollas un análisis profundo que incluye:]

**RESUMEN EJECUTIVO**
[Síntesis general de lo encontrado en 2-3 párrafos]

**HALLAZGOS PRINCIPALES**
[Lista numerada de los descubrimientos más relevantes]

**RAZONAMIENTO ANALÍTICO**
[Explicación detallada de cómo conectaste la información]
- ¿Qué patrones identificaste?
- ¿Cómo se relacionan los documentos entre sí?
- ¿Qué conclusiones se pueden extraer?

**MENSAJES OCULTOS DETECTADOS** 🔐
[Si identificaste codificación o patrones encriptados, explica:]
- Tipo de mensaje oculto encontrado
- Método de encriptación usado
- Cómo se forma el mensaje al unir fragmentos
- Documentos involucrados en la secuencia
- Nivel de confianza en el hallazgo (%)

**CONTEXTO Y SIGNIFICADO**
[Interpretación analítica del conjunto de información]

═══════════════════════════════════════════════════════════
📁 SECCIÓN 2: EVIDENCIAS TEXTUALES CON REFERENCIAS PRECISAS
═══════════════════════════════════════════════════════════

[Para CADA documento fuente, agrupa las citas así:]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 DOCUMENTO: [nombre_exacto_archivo.srt]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 FRAGMENTO #1
⏱️ MARCA TEMPORAL: [MM:SS - MM:SS]
📝 TEXTO LITERAL:
"[Copia exacta del texto del documento, palabra por palabra]"
💡 RELEVANCIA: [Explica por qué este fragmento es importante para tu análisis]

🔹 FRAGMENTO #2
⏱️ MARCA TEMPORAL: [MM:SS - MM:SS]
📝 TEXTO LITERAL:
"[Texto exacto]"
💡 RELEVANCIA: [Explicación]

[Continúa con todos los fragmentos de este documento...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 DOCUMENTO: [siguiente_archivo.srt]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Repite el formato para cada documento usado]

═══════════════════════════════════════════════════════════
📚 SECCIÓN 3: ÍNDICE DE FUENTES CONSULTADAS
═══════════════════════════════════════════════════════════

Total de documentos analizados: [X]

1. [nombre_archivo_1.srt]
   - Fragmentos citados: [X]
   - Temas principales: [lista breve]
   
2. [nombre_archivo_2.srt]
   - Fragmentos citados: [X]
   - Temas principales: [lista breve]

[Continúa con todos los documentos...]

═══════════════════════════════════════════════════════════
🔍 SECCIÓN 4: METADATOS DE ANÁLISIS
═══════════════════════════════════════════════════════════

📊 ESTADÍSTICAS:
- Documentos procesados: [X]
- Fragmentos totales citados: [X]
- Archivos con información relevante: [X]
- Archivos descartados: [X]

🎯 CALIDAD DEL ANÁLISIS:
- Nivel de confianza: [80-100%]
- Precisión temporal estimada: [80-95%]
- Cobertura de la consulta: [Completa/Parcial/Limitada]

🔐 CRIPTOANÁLISIS:
- Mensajes ocultos detectados: [Sí/No]
- Tipo de codificación: [Si aplica]
- Confiabilidad del hallazgo: [%]

⚠️ LIMITACIONES:
- [Lista cualquier limitación encontrada en los datos]
- [Información que falta o está incompleta]
- [Advertencias sobre interpretación]

═══════════════════════════════════════════════════════════


PROTOCOLOS DE RESPUESTA SEGÚN EL CASO
======================================

CASO A: Información Completa Disponible
----------------------------------------
1. Realizar análisis exhaustivo en los 3 niveles
2. Buscar activamente mensajes ocultos
3. Proporcionar respuesta completa con todas las secciones
4. Declarar confianza alta (85-100%)

CASO B: Información Parcial Disponible
---------------------------------------
1. Responder con lo disponible siguiendo el formato completo
2. En SECCIÓN 1, incluir subsección: "INFORMACIÓN NO ENCONTRADA"
3. Listar específicamente qué aspectos de la pregunta no tienen respuesta
4. Sugerir qué documentos adicionales ayudarían
5. Declarar confianza media (60-84%)

CASO C: Sin Información Disponible
-----------------------------------
Responder con este formato exacto:

═══════════════════════════════════════════════════════════
⚠️ ANÁLISIS SIN RESULTADOS
═══════════════════════════════════════════════════════════

Después de analizar exhaustivamente [X] documentos .srt en la base de datos, 
debo informar que NO he encontrado información sobre: [tema consultado]

📊 PROCESO DE BÚSqueda REALIZADO:
- Documentos examinados: [X]
- Palabras clave buscadas: [lista]
- Variantes de términos explorados: [lista]
- Patrones buscados: [descripción]

❌ RESULTADO: No existe evidencia documental que permita responder la consulta.

Como GERARD, tengo prohibido inventar o usar conocimiento externo a los documentos.
Por tanto, no puedo proporcionar una respuesta sin evidencia textual directa.

💡 RECOMENDACIÓN: Verifica si existen documentos adicionales que puedan contener 
esta información o reformula la pregunta con términos alternativos.

═══════════════════════════════════════════════════════════


INSTRUCCIONES ESPECIALES PARA DETECCIÓN DE MENSAJES OCULTOS
============================================================

Cuando analices los documentos, ejecuta SIEMPRE estas verificaciones:

CHECK 1: ANÁLISIS DE INICIALES
-------------------------------
- Extrae la primera letra de oraciones clave en cada documento
- Busca si forman palabras o acrónimos significativos
- Verifica patrones alfabéticos en secuencias temporales

CHECK 2: ANÁLISIS NUMÉRICO
---------------------------
- Observa marcas temporales recurrentes (ej: siempre :33 segundos)
- Identifica números que aparecen repetidamente
- Busca progresiones matemáticas (1,2,3... o 5,10,15...)

CHECK 3: ANÁLISIS DE PALABRAS CLAVE
------------------------------------
- Detecta términos técnicos o inusuales que se repiten
- Marca palabras idénticas en documentos diferentes
- Busca variaciones de un mismo término ("PE", "Proyecto E", "P.E.")

CHECK 4: ANÁLISIS SECUENCIAL
-----------------------------
- Ordena documentos cronológicamente
- Lee fragmentos en ese orden buscando narrativa oculta
- Identifica si hay "capítulos" de una historia mayor

CHECK 5: ANÁLISIS CONTEXTUAL
-----------------------------
- Busca oraciones que solo tienen sentido al juntarlas
- Identifica complementariedades entre documentos
- Detecta información que "falta" deliberadamente

CHECK 6: ANÁLISIS DE ANOMALÍAS
-------------------------------
- Marca frases idénticas en contextos diferentes
- Identifica patrones de lenguaje inusuales
- Busca consistencias sospechosas

Si detectas cualquiera de estos patrones, DEBES reportarlo en la subsección 
"MENSAJES OCULTOS DETECTADOS" con evidencia específica.


ESTILO DE COMUNICACIÓN
======================

Tono: Profesional, analítico, meticuloso, como un investigador forense
Vocabulario: Preciso y técnico cuando sea necesario, pero claro
Actitud: Objetivo, basado en evidencias, transparente sobre el proceso
Formato: Estructurado, organizado, fácil de verificar

SIEMPRE:
✅ Usa conectores lógicos (por tanto, además, sin embargo, en consecuencia)
✅ Numera hallazgos para claridad
✅ Separa claramente opinión analítica de citas textuales
✅ Muestra tu razonamiento paso a paso

NUNCA:
❌ Uses lenguaje vago o ambiguo
❌ Hagas afirmaciones sin respaldo documental
❌ Omitas información contradictoria si existe
❌ Simplifi ques excesivamente análisis complejos


VERIFICACIÓN DE CALIDAD ANTES DE RESPONDER
===========================================

Antes de enviar tu respuesta, verifica:

□ ¿Incluí las 4 secciones obligatorias?
□ ¿Cada cita tiene documento + marca temporal + texto literal?
□ ¿Agrupé las citas por documento fuente?
□ ¿Busqué activamente mensajes ocultos?
□ ¿Mi análisis es profundo e investigativo, no superficial?
□ ¿Declaré mi nivel de confianza?
□ ¿Indiqué claramente qué NO encontré (si aplica)?
□ ¿Todas mis afirmaciones tienen respaldo textual?
□ ¿Respeté las reglas de NO inventar información?
□ ¿Las marcas temporales son lo más precisas posible?

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


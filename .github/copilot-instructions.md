## Propósito breve
Orientaciones rápidas para agentes de código (Copilot/IA) que trabajen en este repositorio. Céntrate en cambios pequeños y seguros: arreglar prompt/formatos, reparar carga de claves, mejorar ingestión o estabilizar la UI de Streamlit.

## Big picture (arquitectura)
- Componentes principales:
  - `ingestar.py`: carga `.srt` desde `documentos_srt/`, los divide y crea la base vectorial Chroma en `./chroma_db`.
  - `consultar_terminal.py`: CLI para interactuar con el agente "GERARD". Usa `GoogleGenerativeAI`, `Chroma` y espera la salida en formato JSON estricto.
  - `consultar_web.py`: UI con Streamlit que reutiliza la misma lógica de recuperación y prompt; añade cache (`@st.cache_resource`/`@st.cache_data`) y registro de conversaciones (`gerard_log.txt`).

## Flujo de datos y por qué está así
- Ingestión -> ChromaDB: `ingestar.py` convierte `.srt` en chunks con `RecursiveCharacterTextSplitter` y crea vectores con `GoogleGenerativeAIEmbeddings` (modelo: `models/embedding-001`). Persistencia en `./chroma_db` para lecturas rápidas.
- Consulta -> Recuperador -> Prompt -> LLM: las apps (`consultar_*`) usan `vectorstore.as_retriever()` para obtener contexto, formatean documentos con `format_docs_with_metadata()` y pasan contexto al `ChatPromptTemplate` que fuerza una salida JSON específica (ver "Formato de salida" abajo).

## Formato de salida requerido (crítico)
- El LLM debe responder estrictamente con un array JSON de objetos; cada objeto tiene `type` y `content`.
  - `type`: `normal` o `emphasis`.
  - `content`: string. Las citas de fuente deben ir dentro del `content` (ej.: `(Fuente: archivo.srt, Timestamp: HH:MM:SS --> HH:MM:SS)`).
- Nota técnica: los scripts extraen el JSON con `re.search(r'\[.*\]', ...)` y luego `json.loads(...)`. Evita texto fuera de ese array o comentarios extra.

## Convenciones y patrones del proyecto
- Prompt/Persona: la personalidad se llama "GERARD". No modifiques la estructura del prompt salvo para correcciones menores; la validación del JSON salida es estricta.
- Limpieza de textos: `get_cleaning_pattern()` elimina etiquetas como `[Spanish (auto-generated)]` o `[DownSub.com]` usando un regex robusto. Reutiliza ese patrón cuando limpies texto.
- Timestamps: hay lógica para eliminar milisegundos `(
  re.sub(r'(\d{2}:\d{2}:\d{2}),\d{3}', r'\1', ...)` antes de mostrar/registrar.
- Logs: conversaciones y respuestas se guardan en `gerard_log.txt` con un intento de convertir el JSON en texto legible (función `get_clean_text_from_json`).

## Dependencias e integración
- Instalación: `pip install -r requirements.txt`. Dependencias clave: `streamlit`, `langchain-google-genai`, `langchain-chroma`, `python-dotenv`, `requests`, `colorama`.
- Model strings: los módulos usan los identificadores `models/gemini-2.5-pro` (LLM) y `models/embedding-001` (embeddings). Cambiar estos nombres requiere validar compatibilidad con el SDK de Google.
- Persistencia Chroma: `./chroma_db` contiene `chroma.sqlite3` y las carpetas de vectores. Evita borrar sin recrear con `ingestar.py`.

## Configuración y secretos
- El código comprueba `os.environ['GOOGLE_API_KEY']` (y también llama `load_dotenv()`), por tanto la forma más fiable es exportar la variable de entorno antes de ejecutar.
- Hay un archivo `.streamlit/secrets.toml` presente en el repo; su contenido contiene la clave, pero actualmente parece estar malformado (cuidado: falta el cierre de comillas). No imprimas ni pegues el valor en commits ni en mensajes.
- Recomendación práctica (PowerShell):
```powershell
$env:GOOGLE_API_KEY = '<YOUR_API_KEY>'
pip install -r requirements.txt
```

## Comandos de desarrollo (rápidos)
- Construir/instalar deps:
```powershell
pip install -r requirements.txt
```
- Crear o recrear la base vectorial:
```powershell
python ingestar.py
```
- Ejecutar CLI interactivo:
```powershell
python consultar_terminal.py
```
- Ejecutar la UI Streamlit:
```powershell
streamlit run consultar_web.py
```

## Problemas comunes y atajos de reparación
- Si Streamlit o los scripts fallan por falta de clave: confirmar `GOOGLE_API_KEY` en entorno o en `.env`; corrige `.streamlit/secrets.toml` si quieres usar ese mecanismo (cerrar la comilla). No confíes que Streamlit exporte automáticamente variables a `os.environ`.
- JSON no válido del LLM: el código busca un array entre corchetes; si el modelo añade texto explicativo fuera del array, extrae solo la parte entre corchetes o ajusta el prompt para reforzar la instrucción (preferible). Mira `consultar_terminal.py` y `consultar_web.py` para ejemplos de cómo validan/parsan.
- Regenerar Chroma: si los resultados de búsqueda parecen vacíos o erráticos, borra `./chroma_db` y vuelve a ejecutar `python ingestar.py`.

## Ejemplos dentro del repo que debes revisar antes de cambiar comportamiento
- `ingestar.py`: uso de `DirectoryLoader(..., loader_cls=TextLoader, loader_kwargs={'encoding':'utf-8'})` — respeta esta configuración al cambiar loaders.
- `consultar_terminal.py`: función `print_json_answer` muestra cómo se colorean `emphasis` y cómo se separan las citas.
- `consultar_web.py`: `format_docs_with_metadata()` limpia nombres de archivo y contenido; la UI aplica `@st.cache_resource` para `load_resources()`.

## Qué no cambiar sin pruebas
- No elimines la regla de salida JSON del prompt (romperá el pipeline de parsing y logging).
- No cambies los nombres de directorio `documentos_srt/` y `chroma_db/` sin actualizar rutas relativas en los tres scripts.

Si algo en estas notas está incompleto o quieres que añada fragmentos de comandos alternativos (Docker, GitHub Actions, tests), dime qué prefieres y lo amplío.

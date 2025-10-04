# GERARD — asistente de búsqueda sobre archivos SRT

Pequeña app que indexa archivos `.srt` en ChromaDB y expone dos frontends:

- `consultar_terminal.py`: CLI interactiva.
- `consultar_web.py`: UI en Streamlit (archivo principal para deploy).

## Contenido del repositorio

- `ingestar.py` — carga `.srt` desde `documentos_srt/`, los divide en trozos y crea `./chroma_db`.
- `consultar_terminal.py` — cliente CLI que pregunta a GERARD y muestra salida JSON coloreada.
- `consultar_web.py` — interfaz Streamlit; espera la variable `GOOGLE_API_KEY` y usa `st.secrets` en despliegue.
- `documentos_srt/` — carpeta con los archivos SRT (datos). No recomendable subir si contienen material privado.
- `chroma_db/` — base vectorial persistida (NO se recomienda subir al repo).
- `.env.sample`, `.streamlit/secrets.toml`, `.github/copilot-instructions.md`, `.gitignore` y `requirements.txt`.

## Requisitos

- Python 3.10+ recomendado
- Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Configuración de la API Key (obligatorio)

La app usa la librería `langchain-google-genai` que requiere `GOOGLE_API_KEY`.

- Localmente: crea un archivo `.env` en la raíz (no comitear) copiando `.env.sample` y añadiendo tu clave:

```powershell
# En PowerShell (temporal para la sesión)
$env:GOOGLE_API_KEY = '<TU_API_KEY>'

# O crea un .env con:
## GOOGLE_API_KEY=<TU_API_KEY>
```

- En Streamlit Cloud: añade la clave desde la interfaz de la App → Settings → Secrets con la entrada exacta `GOOGLE_API_KEY`.

Nota: NO subas esa clave al repositorio. Usa `st.secrets` en Cloud o variables de entorno locales.

## Comandos habituales

- Generar/recrear la base vectorial (ejecutar cuando añadas/actualices SRTs):

```powershell
python ingestar.py
```

- Ejecutar la UI local (Streamlit):

```powershell
streamlit run consultar_web.py
```

- Ejecutar la CLI interactiva:

```powershell
python consultar_terminal.py
```

## Formato de salida (importante)

El modelo debe responder estrictamente con un array JSON de objetos. Cada objeto tiene dos claves:

- `type`: `normal` o `emphasis`.
- `content`: string. Las citas de fuente deben ir dentro del `content`, por ejemplo:

```json
[
  {"type":"normal","content":"Resumen: ... "},
  {"type":"emphasis","content":"concepto clave"},
  {"type":"normal","content":" (Fuente: archivo.srt, Timestamp: 00:01:23 --> 00:01:25)"}
]
```

Los scripts extraen el JSON mediante `re.search(r'\[.*\]')` y luego `json.loads(...)`. Evita que el modelo añada texto fuera del array.

## Seguridad y limpieza de claves expuestas

- Si tu API key estuvo expuesta en commits anteriores, revoca/regenera la clave en Google Cloud inmediatamente.
- Puedo ayudarte a preparar pasos para eliminar la clave del historial Git (BFG o git filter-repo) si lo solicitas.

## Qué subir al repositorio

- Subir: `*.py` (código), `requirements.txt`, `.env.sample`, `.streamlit/secrets.toml` (plantilla), `README.md`, `.github/*`, `.gitignore`.
- No subir: `chroma_db/`, `chroma_db.zip`, `gerard_log.txt`, `.env` con claves, `venv/`, `documentos_srt/` (opcional; subir solo si corresponde).

## Troubleshooting rápido

- Error "GOOGLE_API_KEY no está configurada": asegúrate de que la variable esté en el entorno o añadida en Streamlit Secrets con la clave EXACTA `GOOGLE_API_KEY`.
- Si Chroma devuelve poco o nada: borra `./chroma_db` y ejecuta `python ingestar.py` para regenerarla.

## Próximos pasos sugeridos

- Añadir `README.md` (hecho).
- (Opcional) Crear GitHub Action que bloquee commits con `.env` o `.streamlit/secrets.toml` con claves reales.
- (Opcional) Ayuda para limpiar el historial Git si la clave fue expuesta.

Si quieres que genere la GitHub Action preventiva o que limpie el historial de Git automáticamente, dime y lo preparo.

# 🔑 Explicación de las Dos APIs

## Diagrama Visual

```
┌─────────────────────────────────────────────────────────────────┐
│  STREAMLIT CLOUD - Secrets (lo que necesitas configurar)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  API #1: Google Gemini (LLM)                           │    │
│  │  ────────────────────────────                          │    │
│  │  GOOGLE_API_KEY = "AIzaSy..."                          │    │
│  │                                                         │    │
│  │  ✅ YA LA TIENES (tu app responde preguntas)          │    │
│  │  📝 Solo necesitas copiar el valor que ya existe      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  API #2: Google Service Account (Sheets)              │    │
│  │  ──────────────────────────────────────                │    │
│  │  [gcp_service_account]                                 │    │
│  │  type = "service_account"                              │    │
│  │  project_id = "gerard-logger"                          │    │
│  │  private_key = "-----BEGIN PRIVATE KEY-----..."        │    │
│  │  client_email = "gerard-sheets-logger@..."            │    │
│  │  ...                                                    │    │
│  │                                                         │    │
│  │  ❌ FALTA ESTA (por eso no se registra)              │    │
│  │  📝 Ya está en streamlit_secrets_format.txt           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Flujo de Datos

```
┌──────────────────┐
│  Usuario hace    │
│  una pregunta    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  GERARD usa API #1 (Gemini)                          │
│  ✅ Funciona - genera respuesta                      │
└────────┬─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  GERARD intenta usar API #2 (Sheets Logger)         │
│  ❌ FALLA - no encuentra credenciales               │
│  ⚠️  Por eso no se registra en Google Sheets        │
└──────────────────────────────────────────────────────┘
```

## Respuestas a Preguntas Comunes

### ❓ ¿Son la misma API?
**NO.** Son dos servicios diferentes de Google:
- **API #1:** Google AI Studio / Vertex AI → Para usar Gemini
- **API #2:** Google Service Account → Para acceder a Google Sheets

### ❓ ¿Necesito crear una API nueva?
**NO.** Ya tienes la API de Gemini funcionando. Solo necesitas:
1. Copiar la que ya tienes (API #1)
2. Agregar las credenciales de Sheets (API #2) que ya generé en `streamlit_secrets_format.txt`

### ❓ ¿Dónde está mi API de Gemini actual?
**Opción 1 - En Streamlit Cloud:**
1. https://share.streamlit.io/
2. Tu app → Settings → Secrets
3. Busca la línea `GOOGLE_API_KEY = "..."`
4. Copia ese valor

**Opción 2 - Probar sin ver la actual:**
Si no quieres buscar la actual, puedes generar una nueva:
1. https://aistudio.google.com/app/apikey
2. Click en "Create API Key"
3. Copia la nueva key

### ❓ ¿Qué pasa si uso una API de Gemini diferente?
Nada malo. Puedes usar:
- La misma que ya tienes (recomendado)
- Una nueva API Key del mismo proyecto
- Una API Key de otro proyecto

Todas funcionarán igual mientras tengan acceso a Gemini API.

### ❓ ¿El archivo streamlit_secrets_format.txt ya tiene todo?
**Casi.** Solo necesitas:
1. Abrir `streamlit_secrets_format.txt`
2. Reemplazar `TU_GOOGLE_API_KEY_AQUI` con tu API Key de Gemini
3. Copiar TODO el archivo
4. Pegarlo en Streamlit Cloud → Secrets

## Ejemplo Práctico

### Antes (solo API #1):
```toml
# En Streamlit Cloud → Secrets (estado actual)
GOOGLE_API_KEY = "AIzaSyC1234567890..."
```
✅ GERARD responde preguntas
❌ NO se registra en Google Sheets

### Después (API #1 + API #2):
```toml
# En Streamlit Cloud → Secrets (lo que necesitas)
GOOGLE_API_KEY = "AIzaSyC1234567890..."

[gcp_service_account]
type = "service_account"
project_id = "gerard-logger"
private_key = "-----BEGIN PRIVATE KEY-----..."
client_email = "gerard-sheets-logger@..."
...
```
✅ GERARD responde preguntas
✅ SÍ se registra en Google Sheets

## Resumen Ultra-Simplificado

1. **Abre:** `streamlit_secrets_format.txt`
2. **Busca:** La línea que dice `GOOGLE_API_KEY = "TU_GOOGLE_API_KEY_AQUI"`
3. **Reemplaza:** `TU_GOOGLE_API_KEY_AQUI` con tu API Key actual de Gemini
4. **Copia:** TODO el contenido del archivo
5. **Ve a:** Streamlit Cloud → Tu App → Settings → Secrets
6. **Pega:** Todo el contenido
7. **Save:** Guarda los cambios
8. **¡Listo!** En 2 minutos estará funcionando

---

¿Necesitas ayuda para encontrar tu API Key actual de Gemini?

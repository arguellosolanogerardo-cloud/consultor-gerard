# 🔧 Configurar Google Sheets en Streamlit Cloud

## Problema Actual
Las consultas desde la aplicación web NO se registran en Google Sheets porque **Streamlit Cloud** no tiene acceso a las credenciales de Google.

## Solución: Agregar Credenciales en Streamlit Cloud

### Paso 1: Abrir el archivo de credenciales local
1. Abre el archivo: `e:\proyecto-gemini\google_credentials.json`
2. Copia TODO el contenido (es un archivo JSON grande)

### Paso 2: Configurar Secrets en Streamlit Cloud
1. Ve a tu app en Streamlit Cloud: https://consultor-gerard-x4txzyjv4h3yayhwbhvxea.streamlit.app/
2. Haz clic en el menú (⋮) → **Settings**
3. Ve a la pestaña **Secrets**
4. Pega el siguiente texto (reemplaza con tu JSON real):

```toml
# Google API Key
GOOGLE_API_KEY = "TU_GOOGLE_API_KEY_AQUI"

# Google Sheets Service Account
[gcp_service_account]
type = "service_account"
project_id = "gerard-logger"
private_key_id = "COPIA_DESDE_google_credentials.json"
private_key = "-----BEGIN PRIVATE KEY-----\nCOPIA_LA_CLAVE_PRIVADA_COMPLETA\n-----END PRIVATE KEY-----\n"
client_email = "gerard-sheets-logger@gerard-logger.iam.gserviceaccount.com"
client_id = "COPIA_DESDE_google_credentials.json"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "COPIA_DESDE_google_credentials.json"
```

### Paso 3: Formato Correcto para private_key
⚠️ **MUY IMPORTANTE**: La clave privada debe estar en una sola línea con `\n` en lugar de saltos de línea reales.

**Ejemplo INCORRECTO** (con saltos de línea):
```
private_key = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSj...
...
-----END PRIVATE KEY-----"
```

**Ejemplo CORRECTO** (una sola línea con \n):
```
private_key = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSj...\n...copia_todo_el_contenido...\n-----END PRIVATE KEY-----\n"
```

### Paso 4: Verificar los Valores
Compara con tu archivo `google_credentials.json` y copia estos campos exactamente:
- `project_id`
- `private_key_id`
- `private_key` (en formato de una línea)
- `client_email`
- `client_id`
- `client_x509_cert_url`

### Paso 5: Guardar y Reiniciar
1. Haz clic en **Save**
2. La app se reiniciará automáticamente
3. Las nuevas consultas deberían registrarse en Google Sheets

## Cómo Verificar que Funciona

### Opción 1: Revisar los logs de Streamlit Cloud
1. En tu app → **Manage app** → **Logs**
2. Busca mensajes como:
   - `[INFO] Usando credenciales desde Streamlit secrets`
   - `[OK] Google Sheets Logger conectado exitosamente`
   - `[OK] Interaccion registrada en Google Sheets`

### Opción 2: Hacer una consulta de prueba
1. Abre tu app: https://consultor-gerard-x4txzyjv4h3yayhwbhvxea.streamlit.app/
2. Haz una pregunta cualquiera
3. Ve a Google Sheets: "GERARD - Logs de Usuarios"
4. Verifica que aparezca un nuevo registro

## Solución de Problemas

### Si ves "Google Sheets Logger no disponible"
- Verifica que los paquetes estén en `requirements.txt`:
  ```
  gspread==5.12.0
  oauth2client==4.1.3
  ```

### Si ves "Archivo de credenciales no encontrado"
- Las credenciales NO están configuradas en Streamlit Secrets
- Sigue los pasos anteriores para agregarlas

### Si ves errores de autenticación
- Verifica que el `private_key` esté en formato de una línea
- Verifica que el `client_email` coincida con el Service Account
- Asegúrate de que la hoja "GERARD - Logs de Usuarios" esté compartida con:
  `gerard-sheets-logger@gerard-logger.iam.gserviceaccount.com`

## Archivo secrets.toml Local (Opcional)

Si también quieres que funcione localmente sin usar `google_credentials.json`, crea el archivo:
`.streamlit/secrets.toml` (en tu carpeta local) con el mismo formato.

**NOTA**: Este archivo NO debe subirse a GitHub (ya está en `.gitignore`)

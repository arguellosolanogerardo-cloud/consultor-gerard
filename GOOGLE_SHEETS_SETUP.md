# 📊 Configuración de Google Sheets Logger

Guía paso a paso para ver todos los logs de usuarios de GERARD en Google Sheets en tiempo real.

---

## 🎯 **QUÉ VAS A LOGRAR**

Una hoja de Google Sheets que se actualiza automáticamente con cada consulta de usuarios:

| ID | Fecha/Hora | Usuario | Pregunta | Respuesta | Dispositivo | Navegador | OS | Ciudad | País | IP | Tiempo | Estado |
|----|------------|---------|----------|-----------|-------------|-----------|-------|--------|------|-------|--------|---------|
| abc123 | 2025-10-11 14:30:00 | JUAN | ¿Qué es...? | El amor es... | Desktop | Chrome | Windows | Buenos Aires | AR | 200.x.x.x | 2.34s | ✅ Exitoso |

---

## 📝 **PASO 1: Instalar Dependencias**

```powershell
cd E:\proyecto-gemini
pip install gspread oauth2client
```

Agregar al `requirements.txt`:
```powershell
echo gspread>>requirements.txt
echo oauth2client>>requirements.txt
```

---

## 🔧 **PASO 2: Crear Proyecto en Google Cloud**

### 2.1 Ir a Google Cloud Console

1. Abre: https://console.cloud.google.com/
2. Si no tienes cuenta, créala (es gratis)

### 2.2 Crear Nuevo Proyecto

1. Click en el selector de proyectos (arriba a la izquierda)
2. Click en "NUEVO PROYECTO"
3. Nombre: `GERARD Logger`
4. Click en "CREAR"
5. Espera 10-30 segundos

### 2.3 Seleccionar el Proyecto

1. Click en el selector de proyectos
2. Selecciona "GERARD Logger"

---

## 📡 **PASO 3: Habilitar Google Sheets API**

1. En el menú (☰), ir a: **APIs y servicios** → **Biblioteca**
2. Buscar: `Google Sheets API`
3. Click en "Google Sheets API"
4. Click en "HABILITAR"
5. Esperar a que se habilite

Repetir para **Google Drive API**:
1. Buscar: `Google Drive API`
2. Click en "Google Drive API"
3. Click en "HABILITAR"

---

## 🔑 **PASO 4: Crear Service Account (Credenciales)**

### 4.1 Ir a Credenciales

1. En el menú (☰), ir a: **APIs y servicios** → **Credenciales**
2. Click en "CREAR CREDENCIALES"
3. Seleccionar "Cuenta de servicio"

### 4.2 Configurar Service Account

**Paso 1 de 3:**
- Nombre: `gerard-sheets-logger`
- ID: (se genera automáticamente)
- Descripción: `Service account para logging de GERARD en Google Sheets`
- Click en "CREAR Y CONTINUAR"

**Paso 2 de 3:**
- Función: Seleccionar "Editor" (o puedes dejarlo vacío)
- Click en "CONTINUAR"

**Paso 3 de 3:**
- Dejar vacío
- Click en "LISTO"

### 4.3 Crear Clave JSON

1. En la lista de cuentas de servicio, click en el email de la cuenta recién creada
2. Ir a la pestaña "CLAVES"
3. Click en "AGREGAR CLAVE" → "Crear clave nueva"
4. Tipo: **JSON**
5. Click en "CREAR"
6. **Se descargará automáticamente un archivo JSON** (por ejemplo: `gerard-logger-abc123.json`)

### 4.4 Copiar Email del Service Account

⚠️ **IMPORTANTE:** Copia el email completo que aparece en la lista. Tiene este formato:
```
gerard-sheets-logger@gerard-logger-123456.iam.gserviceaccount.com
```

---

## 📄 **PASO 5: Crear Hoja de Google Sheets**

1. Ir a: https://sheets.google.com/
2. Click en "En blanco" para crear nueva hoja
3. Nombre de la hoja: **GERARD - Logs de Usuarios**
4. Click en "Compartir" (botón azul arriba a la derecha)
5. Pegar el **email del service account** (copiado en paso 4.4)
6. Permisos: **Editor**
7. ⚠️ **DESMARCAR** "Notificar a las personas"
8. Click en "Compartir"

---

## 📦 **PASO 6: Configurar en tu Proyecto**

### 6.1 Renombrar y Mover el Archivo JSON

1. Renombra el archivo descargado a: `google_credentials.json`
2. Muévelo a la carpeta del proyecto: `E:\proyecto-gemini\`

```powershell
# Ejemplo (ajusta la ruta del archivo descargado)
Move-Item "$env:USERPROFILE\Downloads\gerard-logger-*.json" E:\proyecto-gemini\google_credentials.json
```

### 6.2 Agregar al .gitignore

⚠️ **IMPORTANTE:** No subir credenciales a GitHub

```powershell
echo google_credentials.json>>.gitignore
```

### 6.3 Para Streamlit Cloud

1. Ir a: https://share.streamlit.io/
2. Click en tu app "consultor-gerard"
3. Click en "Settings" (⚙️)
4. Ir a "Secrets"
5. Agregar el contenido completo del archivo `google_credentials.json`:

```toml
# Copiar TODO el contenido del archivo google_credentials.json
# incluyendo las llaves { }
```

---

## 🔌 **PASO 7: Integrar en el Código**

El código ya está preparado. Solo necesitas verificar que se active:

```python
# En consultar_web.py, después de init_logger()
from google_sheets_logger import create_sheets_logger

# Crear logger de Google Sheets
sheets_logger = create_sheets_logger()

# Registrar interacción
if sheets_logger:
    sheets_logger.log_interaction(
        interaction_id=interaction_id,
        user=user_name,
        question=question,
        answer=answer,
        device_info=device_info,
        location_info=location_info,
        timing=timing,
        success=True
    )
```

---

## ✅ **PASO 8: Probar**

### 8.1 Probar Localmente

```powershell
cd E:\proyecto-gemini
streamlit run consultar_web.py
```

Haz una pregunta y verifica:
1. La consola debe mostrar: `✅ Google Sheets Logger conectado exitosamente`
2. La hoja de Google Sheets debe actualizarse automáticamente

### 8.2 Verificar en Google Sheets

1. Abre tu hoja: https://sheets.google.com/
2. Busca "GERARD - Logs de Usuarios"
3. Deberías ver la nueva interacción en una fila

---

## 🚀 **DESPLEGAR A STREAMLIT CLOUD**

### Opción A: Con archivo de credenciales

1. Subir `google_credentials.json` a GitHub (⚠️ NO RECOMENDADO por seguridad)

### Opción B: Con Secrets de Streamlit (✅ RECOMENDADO)

1. Copiar el contenido de `google_credentials.json`
2. Ir a Settings → Secrets en Streamlit Cloud
3. Crear un secreto con clave `GOOGLE_SHEETS_CREDENTIALS`
4. Pegar el JSON completo como valor

Modificar el código para leer desde secrets:

```python
import streamlit as st
import json

# En lugar de leer desde archivo
if "GOOGLE_SHEETS_CREDENTIALS" in st.secrets:
    creds_dict = json.loads(st.secrets["GOOGLE_SHEETS_CREDENTIALS"])
    # Guardar temporalmente en archivo
    with open("google_credentials.json", "w") as f:
        json.dump(creds_dict, f)
```

---

## 📊 **VER LOS DATOS**

### En la Hoja de Google Sheets

La hoja se actualiza en tiempo real. Puedes:

- ✅ Ver desde cualquier dispositivo (PC, móvil, tablet)
- ✅ Filtrar por usuario, fecha, país, etc.
- ✅ Crear gráficos y tablas dinámicas
- ✅ Exportar a Excel/CSV
- ✅ Compartir con otras personas
- ✅ Crear fórmulas para análisis

### Columnas Disponibles

| Columna | Descripción |
|---------|-------------|
| ID | Identificador único |
| Fecha/Hora | Timestamp completo |
| Usuario | Nombre del usuario |
| Pregunta | Pregunta completa |
| Respuesta (Resumen) | Primeros 200 caracteres |
| Dispositivo | Desktop/Mobile/Tablet |
| Navegador | Chrome, Firefox, Safari, etc. |
| Sistema Operativo | Windows, macOS, Linux, etc. |
| Ciudad | Ciudad del usuario |
| País | País (código ISO) |
| IP | Dirección IP |
| Tiempo Respuesta | Segundos |
| Estado | ✅ Exitoso / ❌ Error |
| Error | Mensaje de error si aplica |

---

## 🔧 **SOLUCIÓN DE PROBLEMAS**

### Error: "Spreadsheet not found"

- Verifica que la hoja se llame exactamente: `GERARD - Logs de Usuarios`
- Verifica que la compartiste con el email del service account

### Error: "Insufficient permissions"

- Verifica que diste permisos de "Editor" al service account
- Vuelve a compartir la hoja

### No se actualiza la hoja

- Verifica que el archivo `google_credentials.json` esté en la carpeta correcta
- Verifica que la consola muestre "✅ Google Sheets Logger conectado"
- Revisa los permisos de la API en Google Cloud Console

---

## 🎉 **¡LISTO!**

Ahora cada vez que alguien use GERARD desde cualquier parte del mundo, verás la interacción aparecer automáticamente en tu hoja de Google Sheets.

**URL de ejemplo de la hoja:**
```
https://docs.google.com/spreadsheets/d/TU_ID_DE_HOJA/edit
```

---

**Última actualización:** 11 de octubre de 2025

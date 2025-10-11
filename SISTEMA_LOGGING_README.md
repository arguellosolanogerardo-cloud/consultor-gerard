# 📊 Sistema de Logging y Reportes para GERARD

## ✅ **SISTEMA ACTIVADO**

El sistema completo de logging está ahora activo y registrando todas las interacciones.

---

## 📁 **DÓNDE REVISAR LOS DATOS**

### 1. **Logs Diarios (Texto Legible)**
```
📂 logs/
   └── interactions_YYYYMMDD.log
```

**Ejemplo:** `logs/interactions_20251011.log`

**Formato:**
```
================================================================================
[2025-10-11 14:23:45]
Interacción ID: abc123...
Usuario: JUAN
Pregunta: ¿Qué es el amor?
Dispositivo: Desktop - Chrome 119.0 (Windows 10)
Ubicación: Buenos Aires, Argentina (IP: 200.123.45.67)
Tiempo de Respuesta: 2.34 segundos
Estado: ✅ Exitoso
--------------------------------------------------------------------------------
Respuesta:
[contenido de la respuesta...]
================================================================================
```

### 2. **Logs JSON (Para Análisis)**
```
📂 logs/
   └── interactions_YYYYMMDD.json
```

**Formato:** Una interacción por línea en formato JSON
```json
{
  "interaction_id": "abc123...",
  "timestamp": "2025-10-11T14:23:45.123456",
  "user_name": "JUAN",
  "question": "¿Qué es el amor?",
  "answer": "[...]",
  "device": {
    "device_type": "Desktop",
    "browser": "Chrome 119.0",
    "os": "Windows 10",
    "user_agent": "Mozilla/5.0..."
  },
  "location": {
    "ip": "200.123.45.67",
    "city": "Buenos Aires",
    "country": "Argentina",
    "region": "Buenos Aires",
    "latitude": -34.6037,
    "longitude": -58.3816
  },
  "timing": {
    "start_time": "2025-10-11T14:23:45.123456",
    "end_time": "2025-10-11T14:23:47.456789",
    "total_time": 2.333333
  },
  "success": true
}
```

### 3. **Log Simple (Compatible con versión anterior)**
```
📂 proyecto-gemini/
   └── gerard_log.txt
```

---

## 📧 **REPORTES DIARIOS POR EMAIL**

### **Configuración Inicial**

1. **Editar `email_reporter.py`** (líneas 33-39):

```python
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",  # Para Gmail
    "smtp_port": 587,
    "sender_email": "tu_email@gmail.com",  # 👈 TU EMAIL
    "sender_password": "xxxx xxxx xxxx xxxx",  # 👈 CONTRASEÑA DE APLICACIÓN
    "recipient_email": "donde_recibir@email.com",  # 👈 EMAIL DESTINO
    "subject_prefix": "[GERARD] Reporte Diario"
}
```

### **Obtener Contraseña de Aplicación (Gmail)**

1. Ve a: https://myaccount.google.com/security
2. Busca "Contraseñas de aplicaciones"
3. Genera una nueva contraseña para "Mail"
4. Copia la contraseña (formato: `xxxx xxxx xxxx xxxx`)
5. Pégala en `sender_password`

### **Probar el Envío**

```powershell
# Vista previa (genera HTML sin enviar)
python email_reporter.py --preview

# Enviar reporte de ayer
python email_reporter.py

# Enviar reporte de fecha específica
python email_reporter.py --date 20251010
```

### **Programar Envío Automático Diario**

#### **Windows (Task Scheduler)**

1. Abre "Programador de tareas"
2. Crear tarea básica:
   - Nombre: "GERARD Reporte Diario"
   - Desencadenador: Diario a las 8:00 AM
   - Acción: Iniciar programa
     - Programa: `python`
     - Argumentos: `E:\proyecto-gemini\email_reporter.py`
     - Iniciar en: `E:\proyecto-gemini`

#### **Alternativa: Script Automático**

Crear `enviar_reporte_diario.ps1`:
```powershell
cd E:\proyecto-gemini
python email_reporter.py
```

Programarlo con Task Scheduler para ejecutar este script.

---

## 📊 **CONTENIDO DEL REPORTE POR EMAIL**

El reporte HTML incluye:

### **Estadísticas Generales**
- ✅ Total de interacciones
- 👥 Usuarios únicos
- ⏱️ Tiempo promedio de respuesta
- 📈 Tasa de éxito

### **Top 5 Usuarios Más Activos**
| Usuario | Interacciones |
|---------|---------------|
| JUAN    | 15            |
| MARÍA   | 12            |
| PEDRO   | 8             |

### **Dispositivos Utilizados**
| Tipo     | Cantidad |
|----------|----------|
| Desktop  | 25       |
| Mobile   | 10       |
| Tablet   | 2        |

### **Ubicaciones**
| Ciudad, País        | Accesos |
|--------------------|---------|
| Buenos Aires, AR   | 15      |
| Ciudad de México, MX | 12    |
| Madrid, ES         | 10      |

### **Navegadores**
| Navegador  | Uso |
|------------|-----|
| Chrome     | 30  |
| Firefox    | 5   |
| Safari     | 2   |

---

## 🔍 **ANALIZAR DATOS MANUALMENTE**

### **Ver Logs en Tiempo Real**

```powershell
# Ver últimas 20 líneas del log de hoy
Get-Content logs/interactions_20251011.log -Tail 20

# Seguir el log en vivo
Get-Content logs/interactions_20251011.log -Wait
```

### **Buscar Usuario Específico**

```powershell
# Buscar todas las interacciones de JUAN
Select-String -Path "logs/*.log" -Pattern "Usuario: JUAN"
```

### **Contar Interacciones del Día**

```powershell
# Contar líneas que contienen "Interacción ID"
(Get-Content logs/interactions_20251011.log | Select-String "Interacción ID").Count
```

---

## 📌 **OTRAS OPCIONES DE NOTIFICACIÓN**

### **1. Webhook a Slack/Discord**

Modificar `email_reporter.py` para enviar a Slack:
```python
import requests

def send_to_slack(stats):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    message = {
        "text": f"📊 Reporte GERARD: {stats['total_interactions']} interacciones hoy"
    }
    requests.post(webhook_url, json=message)
```

### **2. Guardar en Google Sheets**

Usar API de Google Sheets para exportar automáticamente.

### **3. Dashboard Web**

Crear dashboard con Streamlit/Flask para visualizar en tiempo real.

### **4. Telegram Bot**

Enviar notificaciones vía bot de Telegram.

---

## 🛠️ **MANTENIMIENTO**

### **Rotación de Logs**

Los logs se crean diarios automáticamente. Para limpiar logs antiguos:

```powershell
# Eliminar logs de más de 30 días
Get-ChildItem logs/ -Filter "interactions_*.log" |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
  Remove-Item
```

### **Tamaño de Logs**

Cada log es limitado a 10 MB por defecto. Cuando se alcanza, se crea un nuevo archivo.

---

## ✅ **VERIFICAR QUE ESTÁ FUNCIONANDO**

1. **Hacer una pregunta en la app web**
2. **Revisar el archivo de hoy:**
   ```powershell
   notepad logs/interactions_20251011.log
   ```
3. **Debería aparecer tu interacción con:**
   - Nombre de usuario
   - Dispositivo
   - Ubicación
   - Pregunta y respuesta
   - Tiempo de respuesta

---

## 📞 **SOPORTE**

Si tienes problemas:

1. Verifica que existe la carpeta `logs/`
2. Verifica permisos de escritura
3. Revisa errores en consola de Streamlit
4. Para email: verifica configuración de Gmail

---

## 🎉 **¡SISTEMA LISTO!**

Todo está configurado y funcionando. Solo necesitas:

1. ✅ Usar la aplicación normalmente
2. ✅ Revisar logs en `logs/interactions_YYYYMMDD.log`
3. ✅ Configurar email en `email_reporter.py`
4. ✅ Ejecutar `python email_reporter.py` para recibir reportes

---

**Última actualización:** 11 de octubre de 2025

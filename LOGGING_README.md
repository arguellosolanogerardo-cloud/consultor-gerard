# 📊 Sistema de Logging de Interacciones - GERARD

Sistema completo de registro y análisis de interacciones para la aplicación GERARD.

## 🎯 Características Principales

### ✅ Captura Automática
- **Sin intervención manual**: El sistema captura automáticamente todas las interacciones
- **Timers de alta precisión**: Usa `time.perf_counter()` para mediciones en microsegundos
- **Registro por fases**: Captura tiempos en múltiples etapas del procesamiento

### 📋 Información Capturada

#### Por cada interacción se registra:

1. **Timestamp**: Fecha y hora exacta (formato: YYYY-MM-DD HH:MM:SS)
2. **Usuario**: Nombre de usuario o identificador
3. **Ubicación Geográfica**:
   - País
   - Ciudad  
   - Coordenadas
   - IP del usuario (con opción de anonimización)
4. **Dispositivo**:
   - Tipo (PC, móvil, tablet)
   - Sistema operativo y versión
   - Navegador/versión (web) o Terminal/Shell (terminal)
   - Resolución de pantalla/terminal
5. **Contexto de Acceso**:
   - Plataforma (terminal o web)
   - URL o ruta de acceso
6. **Interacción**:
   - Pregunta completa del usuario
   - Respuesta completa generada
   - Documentos fuente utilizados
7. **Métricas de Tiempo Detalladas**:
   - Tiempo total de procesamiento
   - Tiempo de consulta RAG
   - Tiempo de consulta al LLM
   - Tiempo de post-procesamiento
   - Tiempo de renderizado
   - Latencia total percibida

## 📁 Estructura de Archivos

```
logs/
├── interaction_log_YYYY-MM-DD.txt    # Log legible para humanos
├── interaction_log_YYYY-MM-DD.json   # Log estructurado en JSON
├── error_log_YYYY-MM-DD.txt          # Solo errores
├── performance_summary_YYYY-MM-DD.txt # Resumen estadístico
└── .geo_cache.json                   # Cache de geolocalización
```

### Formato del Registro (TXT)

```
=====================================
REGISTRO #001
Fecha/Hora: 2025-10-06 14:30:45
Usuario: juan_perez
País: Colombia
Ciudad: Bucaramanga
Coordenadas: 7.1301, -73.1259
IP: 192.168.1.100
Dispositivo: PC - Windows 11
Navegador: Chrome 120
Resolución: 1920x1080
Plataforma: Web
URL: http://localhost:8501

PREGUNTA:
¿Cuál es el mensaje principal sobre el maestro Jesús?

RESPUESTA:
[Respuesta completa de GERARD...]

MÉTRICAS DE RENDIMIENTO:
─────────────────────────
⏱️ Tiempo total: 3.450s
⏱️ Tiempo preparación RAG: 0.200s
⏱️ Tiempo consulta LLM: 2.800s
⏱️ Tiempo post-procesamiento: 0.250s
⏱️ Tiempo renderizado: 0.200s
⏱️ Latencia percibida: 3.450s
📊 Tokens procesados: N/A
📄 Documentos recuperados: 8
✅ Estado: Exitoso

=====================================
```

## 🚀 Uso

### Integración Automática

El sistema ya está integrado en:
- ✅ `consultar_terminal.py`
- ✅ `consultar_web.py`

No requiere configuración adicional. El logging se activa automáticamente al ejecutar las aplicaciones.

### Configuración Opcional

```python
from interaction_logger import InteractionLogger

# Crear logger con configuración personalizada
logger = InteractionLogger(
    platform="web",           # "web" o "terminal"
    log_dir="logs",          # Directorio de logs
    anonymize=True,          # Anonimizar IPs y datos sensibles
    max_file_size_mb=10,     # Tamaño máximo antes de rotar
    enable_json=True         # Guardar también en JSON
)
```

### Uso Manual en Código

```python
# 1. Iniciar logging de interacción
session_id = logger.start_interaction(
    user="nombre_usuario",
    question="pregunta del usuario",
    request_info={"user_agent": "...", "url": "..."}
)

# 2. Marcar fases del procesamiento
logger.mark_phase(session_id, "rag_start")
logger.mark_phase(session_id, "llm_start")
# ... procesamiento ...
logger.mark_phase(session_id, "llm_end")

# 3. Registrar respuesta
logger.log_response(session_id, answer, sources, tokens=1250)

# 4. Finalizar logging
logger.end_interaction(session_id, status="success")
```

## 📊 Análisis de Estadísticas

### Uso del Script de Análisis

```bash
# Analizar logs del día actual
python analyze_logs.py

# Analizar una fecha específica
python analyze_logs.py 2025-10-06

# Listar todas las fechas disponibles
python analyze_logs.py --list
```

### Estadísticas Generadas

El script `analyze_logs.py` genera:

1. **Estadísticas Generales**
   - Total de interacciones
   - Tasa de éxito/error
   
2. **Métricas de Rendimiento**
   - Tiempo promedio, mínimo, máximo
   - Percentiles (P50, P95, P99)
   - Desglose por fase (RAG, LLM, etc.)
   
3. **Usuarios Más Activos**
   - Top 10 usuarios por número de consultas
   
4. **Distribución Geográfica**
   - Por país
   - Por ciudad
   
5. **Dispositivos y Plataformas**
   - Distribución web vs terminal
   - Tipos de dispositivos
   - Navegadores más usados
   
6. **Consultas Más Lentas**
   - Top 10 con tiempos detallados
   
7. **Análisis de Errores**
   - Tipos de errores
   - Frecuencia

### Ejemplo de Salida

```
======================================================================
📊 ANÁLISIS DE LOGS - 2025-10-06
======================================================================

📈 ESTADÍSTICAS GENERALES
----------------------------------------------------------------------
Total de interacciones: 45
  ✅ Exitosas: 43 (95.6%)
  ❌ Fallidas: 2 (4.4%)

⚡ MÉTRICAS DE RENDIMIENTO
----------------------------------------------------------------------
Tiempo promedio de respuesta: 3.245s
Tiempo mínimo: 1.234s
Tiempo máximo: 8.901s
Mediana (P50): 2.987s
Percentil 95 (P95): 6.543s
Percentil 99 (P99): 8.234s

Tiempo promedio de LLM: 2.450s (75.5% del total)

👥 USUARIOS MÁS ACTIVOS
----------------------------------------------------------------------
 1. juan_perez: 15 consultas (33.3%)
 2. maria_gomez: 12 consultas (26.7%)
 3. carlos_ruiz: 8 consultas (17.8%)
...
```

## 🔧 Módulos del Sistema

### 1. `interaction_logger.py`
Módulo principal que coordina todo el sistema de logging.

**Características**:
- Gestión de sesiones activas
- Captura de métricas de tiempo
- Rotación automática de archivos
- Generación de resúmenes estadísticos

### 2. `geo_utils.py`
Obtención de información geográfica basada en IP.

**Características**:
- Múltiples APIs de respaldo (ipapi.co, ip-api.com, ipinfo.io)
- Cache local para evitar llamadas repetidas
- Detección de IPs locales/privadas
- Timeout configurable

### 3. `device_detector.py`
Detección de información del dispositivo y sistema.

**Características**:
- Análisis de User-Agent para web
- Detección de sistema operativo y versión
- Identificación de navegadores
- Detección de terminal y shell en entornos de consola

### 4. `analyze_logs.py`
Script para analizar estadísticas de los logs.

## 🔒 Privacidad y Seguridad

### Anonimización de Datos

Para habilitar la anonimización de datos sensibles:

```python
logger = InteractionLogger(anonymize=True)
```

Cuando está habilitada:
- ✅ Las IPs se hashean con SHA-256
- ✅ No se guarda información identificable
- ✅ Los datos geográficos se mantienen agregados

### Exclusión de Logs del Control de Versiones

El archivo `logs/.gitignore` está configurado para:
- ❌ NO versionar archivos `.txt`, `.log`, `.json`
- ❌ NO versionar cache de geolocalización
- ✅ Mantener solo la estructura del directorio

## 📦 Dependencias

Agregadas a `requirements.txt`:
```
requests  # Para APIs de geolocalización
```

Las demás dependencias ya existían en el proyecto.

## 🎨 Personalización

### Modificar Formato de Log

Editar el método `_format_txt_log()` en `interaction_logger.py`:

```python
def _format_txt_log(self, session: Dict[str, Any], counter: int) -> str:
    # Personalizar formato aquí
    pass
```

### Agregar Nuevas Métricas

1. Capturar la métrica con `mark_phase()`:
```python
logger.mark_phase(session_id, "mi_fase_personalizada")
```

2. Calcular en `_calculate_metrics()`:
```python
if "mi_fase" in phases:
    metrics["tiempo_mi_fase"] = phases["mi_fase_end"] - phases["mi_fase_start"]
```

### Rotación de Archivos

Por defecto, los archivos rotan cuando alcanzan 10MB. Para cambiar:

```python
logger = InteractionLogger(max_file_size_mb=50)  # 50MB
```

## 🐛 Troubleshooting

### No se crean archivos de log

**Problema**: No aparecen archivos en el directorio `logs/`

**Soluciones**:
1. Verificar que el directorio existe: `mkdir logs`
2. Comprobar permisos de escritura
3. Revisar `error_log_*.txt` para mensajes de error

### Errores de geolocalización

**Problema**: Aparece "Desconocido" en ubicación geográfica

**Causas comunes**:
- Sin conexión a internet
- IPs locales/privadas
- Límite de rate en APIs gratuitas

**Solución**: El sistema continúa funcionando, solo registra "Desconocido"

### Rendimiento

**Problema**: El logging afecta el rendimiento

**Soluciones**:
1. Desactivar logging JSON: `enable_json=False`
2. Desactivar geolocalización (modificar código)
3. Aumentar intervalo de rotación

## 📞 Soporte

Para reportar problemas o sugerencias:
1. Revisar este documento
2. Comprobar los logs de error
3. Consultar con el equipo de desarrollo

## 🔄 Mantenimiento

### Limpieza de Logs Antiguos

```bash
# Eliminar logs más antiguos de 30 días
find logs/ -name "*.txt" -mtime +30 -delete
find logs/ -name "*.json" -mtime +30 -delete
```

### Backup de Logs

```bash
# Crear backup comprimido
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

## 📝 Changelog

### Versión 1.0.0 (2025-10-06)
- ✨ Implementación inicial del sistema de logging
- ✨ Integración con consultar_web.py y consultar_terminal.py
- ✨ Módulos de geolocalización y detección de dispositivos
- ✨ Script de análisis de estadísticas
- ✨ Documentación completa

---

**Desarrollado para el proyecto GERARD** 🔮

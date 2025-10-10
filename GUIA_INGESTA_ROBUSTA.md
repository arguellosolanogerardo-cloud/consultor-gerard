# 🚀 Sistema Robusto de Ingestión FAISS con Rate Limiting

## 📋 Descripción General

Este sistema optimizado permite crear índices FAISS grandes sin que la API de Google corte el proceso. Incluye:

- ✅ **Rate limiting inteligente** (ventana deslizante)
- ✅ **Reintentos automáticos** con backoff exponencial
- ✅ **Guardado incremental** cada N vectores
- ✅ **Checkpoints** para reanudar procesos interrumpidos
- ✅ **Progress bar** con ETA
- ✅ **Manejo robusto de errores** de API

---

## 🗂️ Archivos del Sistema

### Archivos Principales

1. **`faiss_builder.py`** (331 líneas)
   - Clase `FAISSVectorBuilder`: Motor principal con rate limiting
   - Clase `RateLimiter`: Control de peticiones por minuto
   - Clase `Checkpoint`: Guardado/recuperación de progreso
   - Reintentos automáticos con backoff exponencial

2. **`ingestar_robusto.py`** (240 líneas)
   - Script principal para crear índice FAISS completo
   - Carga archivos .srt, divide en chunks, construye índice
   - Soporta `--force` (recrear) y `--resume` (continuar)

3. **`test_builder.py`** (100 líneas)
   - Script de prueba con subset pequeño (5 archivos)
   - Valida configuración antes de procesar todo

### Archivos Generados

- `faiss_index/index.faiss` - Índice FAISS (vectores)
- `faiss_index/index.pkl` - Documentos (metadatos)
- `faiss_checkpoint.json` - Checkpoint temporal (se elimina al completar)

---

## 🛠️ Instalación de Dependencias

```powershell
# Instalar paquetes necesarios
pip install tenacity tqdm
```

**Nota**: Las demás dependencias ya están en `requirements.txt`

---

## 🚦 Guía de Uso Paso a Paso

### **Paso 1: Verificar API Key**

```powershell
# Verificar que la variable esté configurada
echo $env:GOOGLE_API_KEY
```

Si no aparece nada:

```powershell
$env:GOOGLE_API_KEY = "TU_API_KEY_AQUI"
```

---

### **Paso 2: Prueba Pequeña (RECOMENDADO)**

Antes de procesar todos los documentos, ejecuta una prueba con 5 archivos:

```powershell
python test_builder.py
```

**Qué esperar:**
- Procesará ~50-200 chunks (dependiendo del tamaño de los primeros 5 archivos)
- Tomará ~5-10 minutos
- Creará `faiss_index_test/` con el índice de prueba
- Verás progress bar con ETA

**Si la prueba falla:**
- Revisa la API key
- Verifica conexión a internet
- Chequea logs de error

**Si la prueba funciona:**
- ✅ Puedes proceder al Paso 3

---

### **Paso 3: Crear Índice Completo**

Una vez validada la prueba, crea el índice con TODOS los documentos:

```powershell
python ingestar_robusto.py --force
```

**Parámetros:**
- `--force` : Elimina índice existente y crea uno nuevo
- `--resume` : Continúa desde último checkpoint (si se interrumpió)

**Qué esperar:**
- Procesará ~4000+ chunks (depende de cuántos archivos .srt tengas)
- Tomará **1-2 horas** (con configuración conservadora)
- Progress bar mostrará:
  - Chunks procesados
  - Batch actual/total
  - Total de vectores
  - ETA estimado

**Ejemplo de salida:**

```
======================================================================
🚀 CONSTRUCCIÓN DE ÍNDICE FAISS CON RATE LIMITING ROBUSTO
======================================================================
📄 Total de chunks a procesar: 4109
🔧 Modelo embeddings: models/embedding-001
📐 Dimensión vectorial: 768

⚙️ CONFIGURACIÓN:
   • Rate limit: 50 peticiones/minuto
   • Batch size: 50 documentos/lote
   • Guardar cada: 500 vectores
   • Delay entre lotes: 1.5s
   • Reintentos máximos: 5
   • Backoff exponencial: 2s → 60s

======================================================================
⚡ PROCESAMIENTO DE EMBEDDINGS
======================================================================

Procesando chunks: 100%|██████████| 4109/4109 [1:23:45<00:00, batch=82/82, vectores=4109]

💾 Índice guardado: faiss_index/index.faiss (4109 vectores)
💾 Documentos guardados: faiss_index/index.pkl

======================================================================
✅ ÍNDICE FAISS CREADO EXITOSAMENTE
======================================================================
📁 Ubicación: faiss_index
📄 Archivos:
   • faiss_index/index.faiss
   • faiss_index/index.pkl

📊 Tamaños:
   • index.faiss: 12.45 MB
   • index.pkl: 25.30 MB
   • Total: 37.75 MB
```

---

### **Paso 4: Si el Proceso se Interrumpe**

Si por alguna razón el proceso se detiene (Ctrl+C, error, cierre accidental):

```powershell
python ingestar_robusto.py --resume
```

El sistema:
- ✅ Cargará el checkpoint (`faiss_checkpoint.json`)
- ✅ Restaurará el índice parcial
- ✅ Continuará desde donde quedó
- ✅ NO reprocesará chunks ya procesados

---

## ⚙️ Configuración Avanzada

### Ajustar Rate Limiting

Edita `ingestar_robusto.py`, línea ~130:

```python
config = BuilderConfig(
    rate_limit_per_minute=50,        # ← Cambia este número
    batch_size=50,                   # ← O este
    delay_between_batches=1.5,       # ← O este
    # ...
)
```

**Valores conservadores** (más lento pero más seguro):
```python
rate_limit_per_minute=30
batch_size=25
delay_between_batches=2.5
```

**Valores agresivos** (más rápido pero más riesgoso):
```python
rate_limit_per_minute=60
batch_size=100
delay_between_batches=1.0
```

---

## 🐛 Solución de Problemas

### Error: `GOOGLE_API_KEY not found`

**Causa**: Variable de entorno no configurada

**Solución**:
```powershell
$env:GOOGLE_API_KEY = "TU_CLAVE_AQUI"
```

---

### Error: `Rate limit exceeded (429)`

**Causa**: Demasiadas peticiones por minuto

**Solución**:
1. El sistema reintentará automáticamente con backoff
2. Si persiste, reduce `rate_limit_per_minute` en la config

---

### Error: `KeyboardInterrupt`

**Causa**: Usuario presionó Ctrl+C o timeout largo

**Solución**:
```powershell
# Reanudar desde checkpoint
python ingestar_robusto.py --resume
```

---

### Error: `Connection timeout`

**Causa**: Problema de red o API lenta

**Solución**:
- El sistema reintentará automáticamente (hasta 5 veces)
- Si falla, revisa tu conexión a internet
- Ejecuta `--resume` para continuar

---

### El proceso es MUY lento

**Causa**: Configuración muy conservadora

**Solución**:
1. Aumenta `rate_limit_per_minute` a 60
2. Aumenta `batch_size` a 75
3. Reduce `delay_between_batches` a 1.0

---

### Quiero empezar de cero

**Solución**:
```powershell
# Eliminar índice y checkpoint
Remove-Item -Recurse faiss_index, faiss_checkpoint.json -ErrorAction SilentlyContinue

# Crear nuevo índice
python ingestar_robusto.py --force
```

---

## 📊 Verificación Post-Construcción

Una vez creado el índice, verifica que funciona:

```powershell
python test_faiss.py
```

Deberías ver:
```
✅ Índice cargado correctamente
📊 Número de documentos: 4109
📐 Dimensión de vectores: 768
```

---

## 🔄 Workflow Completo Recomendado

```powershell
# 1. Verificar API key
echo $env:GOOGLE_API_KEY

# 2. Prueba pequeña (5 archivos)
python test_builder.py

# 3. Si prueba OK, crear índice completo
python ingestar_robusto.py --force

# 4. Verificar índice creado
python test_faiss.py

# 5. Ejecutar consultor
streamlit run consultar_web.py
```

---

## ⏱️ Tiempos Estimados

| Chunks | Rate Limit | Tiempo Estimado |
|--------|------------|-----------------|
| 500    | 50 req/min | ~15 minutos     |
| 1000   | 50 req/min | ~30 minutos     |
| 2000   | 50 req/min | ~1 hora         |
| 4000   | 50 req/min | ~2 horas        |

**Nota**: Los tiempos incluyen delays de seguridad. El proceso prioriza **robustez sobre velocidad**.

---

## 📝 Notas Importantes

1. ✅ **El proceso es SEGURO**: Nunca perderás progreso gracias a checkpoints
2. ✅ **Puedes interrumpir**: Ctrl+C guarda el estado automáticamente
3. ✅ **Reanudar es instantáneo**: No reprocesa chunks ya procesados
4. ⚠️ **No modifiques archivos .srt**: Durante el proceso de construcción
5. ⚠️ **Mantén la conexión**: Internet estable es importante

---

## 🎯 Resultado Final

Después de ejecutar exitosamente, tendrás:

```
faiss_index/
  ├── index.faiss      (12-15 MB)  ← Vectores FAISS
  └── index.pkl        (25-30 MB)  ← Documentos originales
```

Este índice:
- ✅ Contiene embeddings de TODOS tus archivos .srt
- ✅ Es compatible con `consultar_web.py` y `consultar_terminal.py`
- ✅ Permite búsquedas semánticas rápidas
- ✅ Incluye metadatos (source, timestamps)

---

## 🚀 Siguiente Paso

Una vez creado el índice:

```powershell
streamlit run consultar_web.py
```

Prueba con: **"dame toda la información de María Magdalena"**

Deberías ver múltiples citas con timestamps de diferentes archivos .srt.

---

## 📞 Troubleshooting Rápido

| Síntoma | Solución |
|---------|----------|
| "API key not found" | `$env:GOOGLE_API_KEY = "TU_CLAVE"` |
| Proceso muy lento | Aumentar `rate_limit_per_minute` |
| Errores 429 frecuentes | Reducir `rate_limit_per_minute` |
| Interrupción accidental | `python ingestar_robusto.py --resume` |
| Quiero empezar de cero | `python ingestar_robusto.py --force` |

---

**¡Buena suerte con la ingestión! 🎉**

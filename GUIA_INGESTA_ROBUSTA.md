# 🚀 Guía de Re-Indexación Optimizada con Protección Google

## 📋 Descripción General

Sistema optimizado para re-crear el índice FAISS con **chunks pequeños** y **protección completa** contra rate limiting de Google.

### ✅ Protecciones Implementadas

- ✅ **Batches pequeños** (50 chunks) para evitar saturar la API
- ✅ **Pausas estratégicas** cada 5 batches (3 segundos)
- ✅ **Retry automático** con espera de 10s en errores puntuales
- ✅ **Retry embeddings** con backoff exponencial (3 intentos: 5s, 10s, 15s)
- ✅ **Guardado parcial** de emergencia si falla
- ✅ **Backup automático** del índice anterior
- ✅ **Task type optimizado** para documentos

### 📊 Configuración Actual

```python
CHUNK_SIZE = 300        # 70% más pequeño que antes (1000)
CHUNK_OVERLAP = 50      # Menos solapamiento (antes 200)
BATCH_SIZE = 50         # Chunks por batch
PAUSE_EVERY = 5         # Pausar cada N batches
PAUSE_SECONDS = 3       # Duración de la pausa
```

### ⏱️ Tiempo Estimado

- **Procesamiento**: ~15-20 minutos
- **Pausas anti-rate-limit**: ~3-5 minutos
- **Overhead**: ~5-10 minutos
- **TOTAL**: **25-35 minutos**

---

## 🗂️ Archivos del Sistema

### Scripts Principales

1. **`reiniciar_indice.py`** ⭐ **RECOMENDADO**
   - Re-indexación optimizada con chunks pequeños (300)
   - Todas las protecciones anti-rate-limit
   - Backup automático del índice anterior
   - Verificación con búsqueda de prueba

2. **`ingestar.py`**
   - Script original mejorado con protecciones
   - Usa chunks grandes (10000)
   - Mismo nivel de protección que reiniciar_indice.py

3. **`check_ready.py`** 🆕
   - Checklist pre-vuelo antes de indexar
   - Verifica API key, dependencias, espacio en disco
   - Test de conexión con Google

### Archivos Generados

- `faiss_index/index.faiss` - Índice FAISS (vectores)
- `faiss_index/index.pkl` - Documentos (metadatos)
- `faiss_index_backup_YYYYMMDD_HHMMSS/` - Backup del índice anterior
- `faiss_index_parcial/` - Guardado de emergencia (si falla)

---

## 🛠️ Instalación de Dependencias

```powershell
# Verificar que todo está instalado
pip install -r requirements.txt
```

**Dependencias críticas**:
- `langchain-google-genai` - Embeddings y LLM
- `langchain-community` - FAISS vectorstore
- `faiss-cpu` - Motor de vectores
- `python-dotenv` - Variables de entorno

---

## 🚦 Guía de Uso Paso a Paso

### **Paso 0: Checklist Pre-Vuelo** 🆕

Antes de empezar, verifica que todo está listo:

```powershell
python check_ready.py
```

**Este script verifica:**
1. ✅ API Key de Google configurada
2. ✅ Directorio `documentos_srt/` con archivos .srt
3. ✅ Espacio libre en disco (>1GB)
4. ✅ Dependencias Python instaladas
5. ✅ Script `reiniciar_indice.py` presente y actualizado
6. ✅ Índice actual (si existe)
7. ✅ Test de conexión con Google Generative AI

**Resultado esperado:**
```
✅ ¡TODO LISTO PARA LA INDEXACIÓN!

Puedes ejecutar:
   python reiniciar_indice.py
```

Si hay errores, el script te dirá exactamente qué corregir.

---

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

### **Paso 2: Re-Indexar con Chunks Pequeños** ⭐ **RECOMENDADO**

Ejecuta el script optimizado con todas las protecciones:

```powershell
python reiniciar_indice.py
```

**Qué hace este script:**
1. 🔒 Crea backup automático del índice anterior
2. 📂 Carga todos los archivos .srt
3. ✂️ Divide en chunks de 300 caracteres (70% más pequeños)
4. 🧠 Crea embeddings en batches de 50 con pausas estratégicas
5. 💾 Construye y guarda el índice FAISS
6. ✅ Verifica con búsqueda de prueba

**Qué esperar:**
```
╔══════════════════════════════════════════════════════════╗
║        RE-INDEXACIÓN OPTIMIZADA - CHUNKS PEQUEÑOS        ║
╚══════════════════════════════════════════════════════════╝

📦 Chunk size: 300 (antes: 1000) - 70% más pequeño
🔗 Overlap: 50 (antes: 200)
📂 Directorio: documentos_srt
🎯 Índice: faiss_index

1️⃣  BACKUP DEL ÍNDICE ANTERIOR
✅ Backup: faiss_index_backup_20251010_220000
✅ Índice anterior eliminado

2️⃣  CARGANDO ARCHIVOS .SRT
✅ 200 archivos cargados
   1,500,000 caracteres totales

3️⃣  DIVIDIENDO EN CHUNKS PEQUEÑOS
✅ 12,345 chunks creados
   62 chunks por documento (promedio)
   Tamaño promedio: 280 caracteres
   Rango: 100 - 300 caracteres

4️⃣  INICIALIZANDO EMBEDDINGS CON RETRY
✅ Embeddings de Google listos

5️⃣  CREANDO ÍNDICE FAISS CON PROTECCIÓN ANTI-RATE-LIMIT
⏳ Procesando en batches con pausas estratégicas...
ℹ️ Pausas cada 5 batches para evitar cortes de Google

   Batch 1/247 (50 chunks)... ✅
   Batch 2/247 (50 chunks)... ✅
   Batch 3/247 (50 chunks)... ✅
   Batch 4/247 (50 chunks)... ✅
   Batch 5/247 (50 chunks)... ✅
   💤 Pausa de 3s (evitar rate limit)...
   Batch 6/247 (50 chunks)... ✅
   ...

✅ Índice FAISS creado: 12,345 chunks

6️⃣  GUARDANDO ÍNDICE
✅ Índice guardado: faiss_index
   Tamaño: 45.23 MB

7️⃣  VERIFICACIÓN
✅ Índice verificado: 12,345 documentos

🧪 PRUEBA DE BÚSQUEDA:
   Query: 'linaje ra tric jac bis'
   Resultados: 5
   
   Top resultado:
   • Score: 0.6234
   • Fuente: DESCUBRIENDO...
   ✅ ¡Encuentra el documento correcto!

📊 ESTADÍSTICAS:
   • Archivos: 200
   • Chunks: 12,345 (antes: ~4,109)
   • Chunk size: 300 caracteres (antes: 1000)
   • Tamaño índice: 45.23 MB
   • Backup: faiss_index_backup_20251010_220000

🎯 MEJORAS:
   ✓ Chunks 70% más pequeños (1000→300)
   ✓ Mayor precisión en búsquedas
   ✓ Menos dilución semántica
   ✓ k=25 en consultar_web.py
   ✓ Protección anti-rate-limit de Google
   ✓ Retry automático en errores
   ✓ Guardado parcial si falla
```

**Tiempo estimado**: 25-35 minutos

---

### **Paso 3: Reiniciar Streamlit**

Una vez completada la re-indexación:

```powershell
# Detener Streamlit si está corriendo
Get-Process | Where-Object {$_.ProcessName -eq "streamlit"} | Stop-Process -Force

# Reiniciar con el nuevo índice
streamlit run consultar_web.py
```

**Verificación:**
- Prueba una búsqueda que antes no funcionaba: "linaje ra tric jac bis"
- Deberías obtener resultados relevantes ahora

---

### **Alternativa: Script Original Mejorado**

Si prefieres usar el script original con chunks grandes pero protegido:

```powershell
python ingestar.py --force

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

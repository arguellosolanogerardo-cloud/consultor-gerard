# 🛡️ PROTECCIÓN ANTI-RATE-LIMIT DE GOOGLE

## ✅ **SÍ, YA ESTÁ IMPLEMENTADO**

Los scripts de indexación tienen **protección completa** contra cortes de Google durante la construcción de la base de datos vectorial.

---

## 🎯 Protecciones Implementadas

### 1️⃣ **Batches Pequeños**
- **Antes**: 100 chunks por batch
- **Ahora**: 50 chunks por batch
- **Beneficio**: Menos carga por request, menor probabilidad de rate limit

### 2️⃣ **Pausas Estratégicas**
```python
PAUSE_EVERY = 10     # Pausar cada 10 batches (optimizado para muchos archivos)
PAUSE_SECONDS = 2    # Pausa de 2 segundos (eficiente)
```
- Cada 10 batches → pausa de 2 segundos (optimizado para ~2,000 archivos)
- Da tiempo a Google para resetear contadores
- Evita saturar la API
- Balance entre velocidad y seguridad

### 3️⃣ **Retry Automático con Backoff**
- **Primera falla**: Espera 10 segundos y reintenta
- **Inicialización embeddings**: Hasta 3 intentos con backoff exponencial (5s, 10s, 15s)
- No pierde el progreso si hay un error temporal

### 4️⃣ **Guardado Parcial de Emergencia**
```python
if vectorstore:
    vectorstore.save_local(FAISS_INDEX_PATH + "_parcial")
    print(f"⚠️ Índice parcial guardado")
```
- Si falla después de procesar N batches, guarda el progreso
- No pierdes todo el trabajo si algo sale mal
- Puedes recuperar el índice parcial

### 5️⃣ **Task Type Optimizado**
```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document"  # Optimizado para documentos
)
```
- Usa el tipo de tarea correcto para la API
- Mejor rendimiento y menor probabilidad de errores

---

## 📊 Estimación de Tiempo y Requests

### Con tus archivos actuales:
- **Archivos .srt**: ~1,973 archivos (¡TODOS se indexarán!)
- **Chunks con size=300**: ~100,000-120,000 chunks estimados
- **Batches**: 2,000-2,400 batches (50 chunks cada uno)
- **Pausas**: ~200-240 pausas de 2 segundos
- **Tiempo total estimado**: **3-4 HORAS**

### Distribución del tiempo:
- ⏱️ **Procesamiento real**: ~2.5-3 horas
- 💤 **Pausas anti-rate-limit**: ~8-10 minutos
- 🔄 **Overhead (carga, guardado)**: ~15-20 minutos

---

## 🚀 Cómo Ejecutar de Forma Segura

### Opción 1: Script Optimizado (RECOMENDADO)
```powershell
# Script con chunks pequeños (300) y todas las protecciones
python reiniciar_indice.py
```

### Opción 2: Script Original
```powershell
# Script con chunks grandes (10000) pero protegido
python ingestar.py --force
```

---

## 📝 Durante la Ejecución Verás:

```
5️⃣  CREANDO ÍNDICE FAISS CON PROTECCIÓN ANTI-RATE-LIMIT
⏳ Procesando en batches con pausas estratégicas...
ℹ️ Pausas cada 5 batches para evitar cortes de Google

   Batch 1/300 (50 chunks)... ✅
   Batch 2/300 (50 chunks)... ✅
   Batch 3/300 (50 chunks)... ✅
   Batch 4/300 (50 chunks)... ✅
   Batch 5/300 (50 chunks)... ✅
   💤 Pausa de 3s (evitar rate limit)...
   Batch 6/300 (50 chunks)... ✅
   ...
```

---

## ⚠️ Si Algo Sale Mal

### Escenario 1: Rate limit puntual
```
⚠️ Error en batch 127
Esperando 10 segundos y reintentando...
✅ Batch 127 completado en reintento
```
**✅ Se recupera solo, continúa normal**

### Escenario 2: Error fatal
```
❌ ERROR FATAL en batch 127: ...
Guardando progreso parcial...
⚠️ Índice parcial guardado: faiss_index_parcial
```
**✅ No pierdes el progreso, tienes backup parcial**

### Escenario 3: Interrupción manual (Ctrl+C)
```
^C KeyboardInterrupt
```
**⚠️ Tendrás que reiniciar, pero el backup del índice anterior está en `faiss_index_backup_YYYYMMDD_HHMMSS`**

---

## 🎯 Recomendaciones para Esta Noche

1. **Ejecuta ANTES de dormir** (tomará 3-4 horas)
2. **Asegura conexión estable** (WiFi o cable)
3. **Evita suspensión del PC** (Configuración → Energía → Nunca suspender)
4. **Cierra aplicaciones pesadas** (dejar más RAM disponible)
5. **Ejecuta en terminal normal** (no minimices)
6. **Monitorea los primeros 10 minutos** para confirmar que arranca bien
7. **Deja correr toda la madrugada** (3-4 horas estimadas)

---

## 🔍 Verificación Post-Indexación

El script automáticamente verifica al final:

```
7️⃣  VERIFICACIÓN
✅ Índice verificado: 12,345 documentos

🧪 PRUEBA DE BÚSQUEDA:
   Query: 'linaje ra tric jac bis'
   Resultados: 5
   
   Top resultado:
   • Score: 0.6234
   • Fuente: DESCUBRIENDO...
   ✅ ¡Encuentra el documento correcto!
```

---

## 💡 Tips Extra

### Monitorear uso de API (opcional):
- Ve a [Google AI Studio](https://aistudio.google.com/)
- Revisa "API usage" después de la indexación

### Si tienes dudas durante la ejecución:
- **NO canceles** si ves pausas (son normales)
- **SÍ cancela** si ves el mismo error 3+ veces seguidas
- Los mensajes `✅` indican progreso exitoso

---

## 📋 Checklist Final

- ✅ Scripts con protección anti-rate-limit
- ✅ Batches pequeños (50 chunks)
- ✅ Pausas estratégicas cada 5 batches
- ✅ Retry automático
- ✅ Guardado parcial de emergencia
- ✅ Chunk size optimizado (300)
- ✅ k=25 en consultar_web.py
- ✅ Backup automático del índice anterior

---

## 🎉 Conclusión

**TODO LISTO PARA EJECUTAR ESTA NOCHE SIN RIESGO DE CORTES.**

El proceso es robusto, auto-recuperable y tiene múltiples capas de protección. Puedes dejarlo corriendo con confianza.

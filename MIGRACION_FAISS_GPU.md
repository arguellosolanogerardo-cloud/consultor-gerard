# 🚀 GUÍA DE MIGRACIÓN: FAISS-CPU → FAISS-GPU

**Fecha**: 11 de octubre de 2025  
**Hardware**: RTX 3060 Ti (8 GB VRAM, 4864 CUDA cores)  
**Índice actual**: 84,266 documentos, 317 MB

---

## 📊 ANÁLISIS DE RENDIMIENTO

### Tiempo de búsqueda actual (FAISS-CPU):
- **Búsqueda vectorial (k=100)**: 100-200ms
- **Búsqueda keyword**: 50-150ms
- **Total búsqueda**: 150-350ms
- **LLM (cuello de botella)**: 2000-5000ms
- **TOTAL query**: 2.2-5.5 segundos

### Tiempo esperado con FAISS-GPU:
- **Búsqueda vectorial (k=100)**: 5-20ms ⚡ (10-20x más rápido)
- **Búsqueda keyword**: 50-150ms (sin cambios)
- **Total búsqueda**: 55-170ms
- **LLM (cuello de botella)**: 2000-5000ms
- **TOTAL query**: 2.1-5.2 segundos

**MEJORA NETA**: ~100-300ms (3-7% más rápido)

---

## ✅ VENTAJAS

1. **Velocidad**: Búsqueda vectorial 10-20x más rápida
2. **Escalabilidad**: Con 500K+ docs, diferencia sería dramática
3. **Paralelización**: RTX 3060 Ti tiene 4,864 CUDA cores
4. **Menor uso CPU**: Libera procesador para otras tareas

## ❌ DESVENTAJAS

1. **Complejidad**: Requiere CUDA Toolkit + drivers
2. **VRAM**: ~500 MB ocupados (de 8 GB disponibles)
3. **Mejora marginal**: LLM sigue siendo cuello de botella (80-90%)
4. **Keyword search**: Sigue en CPU (no se acelera)

---

## 🔧 INSTALACIÓN

### 1. Verificar CUDA

```powershell
# Verificar GPU
nvidia-smi

# Verificar CUDA instalado
nvcc --version
```

Si CUDA no está instalado:
1. Descargar de: https://developer.nvidia.com/cuda-downloads
2. Instalar CUDA Toolkit 12.x
3. Reiniciar sistema

### 2. Instalar FAISS-GPU

```powershell
# Desinstalar faiss-cpu
pip uninstall faiss-cpu

# Instalar faiss-gpu
pip install faiss-gpu

# Verificar instalación
python -c "import faiss; print(f'GPUs detectadas: {faiss.get_num_gpus()}')"
# Debe mostrar: GPUs detectadas: 1
```

### 3. Modificar `load_resources()` en `consultar_web.py`

**ANTES (CPU)**:
```python
@st.cache_resource
def load_resources():
    # ... código de embeddings ...
    
    with st.spinner('Cargando índice FAISS desde disco...'):
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"[DEBUG load_resources] FAISS cargado exitosamente con {vectorstore.index.ntotal} documentos")
    
    return llm, vectorstore
```

**DESPUÉS (GPU)**:
```python
import faiss

@st.cache_resource
def load_resources():
    # ... código de embeddings ...
    
    with st.spinner('Cargando índice FAISS desde disco...'):
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"[DEBUG load_resources] FAISS cargado exitosamente con {vectorstore.index.ntotal} documentos")
        
        # Migrar índice a GPU
        try:
            num_gpus = faiss.get_num_gpus()
            if num_gpus > 0:
                print(f"[DEBUG] Migrando índice a GPU (detectadas {num_gpus} GPUs)...")
                res = faiss.StandardGpuResources()  # Recursos GPU
                gpu_index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)  # GPU 0
                vectorstore.index = gpu_index
                print(f"[DEBUG] ✅ Índice migrado a GPU 0 (RTX 3060 Ti)")
            else:
                print(f"[DEBUG] ⚠️ No se detectaron GPUs, usando CPU")
        except Exception as e:
            print(f"[DEBUG] ⚠️ Error migrando a GPU, usando CPU: {e}")
    
    return llm, vectorstore
```

### 4. Reiniciar aplicación

```powershell
Get-Process | Where-Object {$_.ProcessName -eq "streamlit"} | Stop-Process -Force
streamlit run consultar_web.py
```

### 5. Verificar en logs

Deberías ver:
```
[DEBUG] Migrando índice a GPU (detectadas 1 GPUs)...
[DEBUG] ✅ Índice migrado a GPU 0 (RTX 3060 Ti)
```

---

## 📊 BENCHMARKING

### Script de prueba:

```python
# test_gpu_performance.py
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY no configurada")

# Cargar índice
print("Cargando índice...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Test CPU
print("\n=== BENCHMARK CPU ===")
query = "informacion sobre linaje ra bis jac tric"
times_cpu = []
for i in range(10):
    start = time.perf_counter()
    vectorstore.similarity_search(query, k=100)
    elapsed = (time.perf_counter() - start) * 1000
    times_cpu.append(elapsed)
    print(f"Query {i+1}: {elapsed:.2f}ms")

avg_cpu = sum(times_cpu) / len(times_cpu)
print(f"CPU PROMEDIO: {avg_cpu:.2f}ms")

# Migrar a GPU
print("\n=== MIGRANDO A GPU ===")
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
    vectorstore.index = gpu_index
    print("✅ Migrado a GPU")
    
    # Test GPU
    print("\n=== BENCHMARK GPU ===")
    times_gpu = []
    for i in range(10):
        start = time.perf_counter()
        vectorstore.similarity_search(query, k=100)
        elapsed = (time.perf_counter() - start) * 1000
        times_gpu.append(elapsed)
        print(f"Query {i+1}: {elapsed:.2f}ms")
    
    avg_gpu = sum(times_gpu) / len(times_gpu)
    print(f"GPU PROMEDIO: {avg_gpu:.2f}ms")
    
    # Comparación
    print(f"\n=== COMPARACIÓN ===")
    print(f"CPU: {avg_cpu:.2f}ms")
    print(f"GPU: {avg_gpu:.2f}ms")
    print(f"MEJORA: {(avg_cpu / avg_gpu):.1f}x más rápido")
else:
    print("❌ No se detectaron GPUs")
```

Ejecutar:
```powershell
python test_gpu_performance.py
```

Resultado esperado:
```
CPU PROMEDIO: 150.23ms
GPU PROMEDIO: 12.45ms
MEJORA: 12.1x más rápido
```

---

## ⚠️ CONSIDERACIONES

### Uso de VRAM:
- **Índice actual**: ~500 MB en VRAM
- **VRAM disponible**: 8 GB (RTX 3060 Ti)
- **Margen**: 7.5 GB libres (suficiente)

### Compatibilidad:
- ✅ Windows 11
- ✅ RTX 3060 Ti (CUDA Compute 8.6)
- ✅ Python 3.13
- ⚠️ Requiere CUDA 11.x o 12.x

### Rollback:
Si algo falla, volver a CPU:
```powershell
pip uninstall faiss-gpu
pip install faiss-cpu
# Revertir cambios en consultar_web.py
```

---

## 🎯 RECOMENDACIÓN

### **NO migrar ahora SI**:
- ✅ El tiempo de respuesta actual (2-6s) es aceptable
- ✅ Tienes < 100K documentos
- ✅ Priorizas estabilidad sobre velocidad

### **SÍ migrar SI**:
- 📈 Planeas escalar a 500K+ documentos
- ⚡ Necesitas respuestas < 1 segundo
- 🔄 Harás búsquedas batch (100+ simultáneas)
- 🧪 Quieres experimentar con optimizaciones

---

## 📝 PRÓXIMOS PASOS

### Opción 1: Migrar ahora
1. Instalar CUDA Toolkit
2. `pip install faiss-gpu`
3. Modificar `consultar_web.py`
4. Ejecutar benchmark
5. Commit cambios

### Opción 2: Migrar después (recomendado)
1. **Validar** que búsqueda híbrida funciona bien
2. **Monitorear** tiempos de respuesta reales
3. **Decidir** si 100-200ms es un problema
4. **Migrar** solo si es necesario

---

## 🔗 RECURSOS

- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- FAISS-GPU docs: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
- Benchmark tools: https://github.com/facebookresearch/faiss/tree/main/benchs

---

**Conclusión**: La migración a GPU es **técnicamente viable** pero **no urgente**. El LLM (2-5s) sigue siendo el cuello de botella, no la búsqueda (100-200ms). Prioriza primero **validar la búsqueda híbrida** y migra a GPU solo si necesitas escalar o reducir tiempos < 1 segundo.

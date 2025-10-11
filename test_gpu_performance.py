"""
Script de benchmark: FAISS-CPU vs FAISS-GPU

Compara tiempos de búsqueda vectorial en CPU vs GPU
para el índice FAISS del proyecto consultor-gerard.

Uso:
    python test_gpu_performance.py

Requiere:
    - faiss-gpu instalado (pip install faiss-gpu)
    - CUDA Toolkit instalado
    - GPU compatible (RTX 3060 Ti en este caso)
"""
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Intentar importar faiss (funciona con faiss-cpu o faiss-gpu)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("❌ ERROR: faiss no está instalado")
    print("   Instalar con: pip install faiss-cpu  (o faiss-gpu)")
    exit(1)

# Cargar variables de entorno
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY no está configurada")

print("=" * 80)
print("🔬 BENCHMARK: FAISS-CPU vs FAISS-GPU")
print("=" * 80)

# Cargar índice
print("\n📂 Cargando índice FAISS...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

num_docs = vectorstore.index.ntotal
print(f"✅ Índice cargado: {num_docs:,} documentos")

# Queries de prueba
test_queries = [
    "informacion sobre linaje ra bis jac tric",
    "que es la quinta dimension",
    "quien es el maestro jesus",
    "explicacion de la atlantida",
    "como funciona la reencarnacion"
]

k_values = [40, 100]  # Probar con k=40 y k=100

# Test CPU
print("\n" + "=" * 80)
print("⚙️  BENCHMARK CPU")
print("=" * 80)

results_cpu = {}

for k in k_values:
    print(f"\n🔍 Probando con k={k}...")
    times_cpu = []
    
    for i, query in enumerate(test_queries, 1):
        start = time.perf_counter()
        vectorstore.similarity_search(query, k=k)
        elapsed = (time.perf_counter() - start) * 1000
        times_cpu.append(elapsed)
        print(f"  Query {i}: {elapsed:6.2f}ms - '{query[:40]}...'")
    
    avg_cpu = sum(times_cpu) / len(times_cpu)
    results_cpu[k] = avg_cpu
    print(f"  PROMEDIO (k={k}): {avg_cpu:6.2f}ms")

# Verificar si hay GPU disponible
num_gpus = faiss.get_num_gpus()
print(f"\n🖥️  GPUs detectadas: {num_gpus}")

if num_gpus == 0:
    print("\n⚠️  NO se detectaron GPUs CUDA compatibles")
    print("   Verifica:")
    print("   1. CUDA Toolkit instalado (https://developer.nvidia.com/cuda-downloads)")
    print("   2. Drivers NVIDIA actualizados")
    print("   3. faiss-gpu instalado (pip install faiss-gpu)")
    print("\n📊 RESUMEN CPU:")
    for k, avg in results_cpu.items():
        print(f"   k={k:3d}: {avg:6.2f}ms")
    exit(0)

# Migrar a GPU
print(f"\n🚀 Migrando índice a GPU 0...")
try:
    res = faiss.StandardGpuResources()
    
    # Configurar recursos GPU
    # Puedes ajustar memoria temporal (por defecto usa lo que necesite)
    # res.setTempMemory(256 * 1024 * 1024)  # 256 MB temp memory
    
    gpu_index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
    vectorstore.index = gpu_index
    print("✅ Índice migrado a GPU 0")
except Exception as e:
    print(f"❌ Error migrando a GPU: {e}")
    print("\n📊 RESUMEN CPU:")
    for k, avg in results_cpu.items():
        print(f"   k={k:3d}: {avg:6.2f}ms")
    exit(1)

# Test GPU
print("\n" + "=" * 80)
print("🚀 BENCHMARK GPU")
print("=" * 80)

results_gpu = {}

for k in k_values:
    print(f"\n🔍 Probando con k={k}...")
    times_gpu = []
    
    for i, query in enumerate(test_queries, 1):
        start = time.perf_counter()
        vectorstore.similarity_search(query, k=k)
        elapsed = (time.perf_counter() - start) * 1000
        times_gpu.append(elapsed)
        print(f"  Query {i}: {elapsed:6.2f}ms - '{query[:40]}...'")
    
    avg_gpu = sum(times_gpu) / len(times_gpu)
    results_gpu[k] = avg_gpu
    print(f"  PROMEDIO (k={k}): {avg_gpu:6.2f}ms")

# Comparación
print("\n" + "=" * 80)
print("📊 COMPARACIÓN FINAL")
print("=" * 80)

for k in k_values:
    cpu_time = results_cpu[k]
    gpu_time = results_gpu[k]
    speedup = cpu_time / gpu_time
    improvement = ((cpu_time - gpu_time) / cpu_time) * 100
    
    print(f"\n🔢 k = {k}")
    print(f"   CPU: {cpu_time:6.2f}ms")
    print(f"   GPU: {gpu_time:6.2f}ms")
    print(f"   SPEEDUP: {speedup:.1f}x más rápido")
    print(f"   MEJORA: {improvement:.1f}% reducción de tiempo")

# Análisis de impacto en tiempo total de query
print("\n" + "=" * 80)
print("⏱️  IMPACTO EN TIEMPO TOTAL DE QUERY")
print("=" * 80)

# Tiempos típicos de otros componentes
embedding_time = 75  # 50-100ms promedio
keyword_search_time = 100  # 50-150ms promedio
llm_time = 3500  # 2000-5000ms promedio

for k in k_values:
    cpu_total = embedding_time + results_cpu[k] + keyword_search_time + llm_time
    gpu_total = embedding_time + results_gpu[k] + keyword_search_time + llm_time
    total_improvement = cpu_total - gpu_total
    total_improvement_pct = (total_improvement / cpu_total) * 100
    
    print(f"\n🔢 k = {k}")
    print(f"   Componentes:")
    print(f"     - Embedding query:     {embedding_time:6.0f}ms")
    print(f"     - Búsqueda vectorial:  {results_cpu[k]:6.2f}ms (CPU) → {results_gpu[k]:6.2f}ms (GPU)")
    print(f"     - Búsqueda keyword:    {keyword_search_time:6.0f}ms")
    print(f"     - LLM generación:      {llm_time:6.0f}ms")
    print(f"   TOTAL:")
    print(f"     CPU: {cpu_total:6.0f}ms ({cpu_total/1000:.2f}s)")
    print(f"     GPU: {gpu_total:6.0f}ms ({gpu_total/1000:.2f}s)")
    print(f"   MEJORA NETA: {total_improvement:6.0f}ms ({total_improvement_pct:.1f}%)")

print("\n" + "=" * 80)
print("💡 CONCLUSIÓN")
print("=" * 80)

avg_speedup = sum(results_cpu[k] / results_gpu[k] for k in k_values) / len(k_values)
print(f"   Speedup promedio: {avg_speedup:.1f}x")

if avg_speedup >= 10:
    print("   ✅ GPU ofrece EXCELENTE mejora (>10x)")
    print("   🎯 RECOMENDACIÓN: Migrar a GPU es muy beneficioso")
elif avg_speedup >= 5:
    print("   ✅ GPU ofrece BUENA mejora (5-10x)")
    print("   🎯 RECOMENDACIÓN: Migrar a GPU es beneficioso")
elif avg_speedup >= 2:
    print("   ⚠️  GPU ofrece mejora moderada (2-5x)")
    print("   🎯 RECOMENDACIÓN: Migrar a GPU solo si planeas escalar")
else:
    print("   ⚠️  GPU ofrece mejora marginal (<2x)")
    print("   🎯 RECOMENDACIÓN: Mantener CPU es suficiente")

# Calcular mejora en tiempo total
total_improvement_avg = sum(
    ((embedding_time + results_cpu[k] + keyword_search_time + llm_time) -
     (embedding_time + results_gpu[k] + keyword_search_time + llm_time))
    for k in k_values
) / len(k_values)

print(f"\n   Mejora en tiempo total de query: ~{total_improvement_avg:.0f}ms")

if total_improvement_avg < 200:
    print("   ⚠️  LLM sigue siendo el cuello de botella principal")
    print("   💡 Para queries más rápidas, considera:")
    print("      - Reducir temperature del LLM")
    print("      - Usar modelo más pequeño (si disponible)")
    print("      - Implementar caché de respuestas frecuentes")

print("\n" + "=" * 80)

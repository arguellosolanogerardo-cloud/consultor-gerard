#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CHECKLIST PRE-INDEXACIÓN
========================
Verifica que todo está listo para ejecutar reiniciar_indice.py de forma segura.
"""

import os
import sys
from dotenv import load_dotenv

print("""
╔═══════════════════════════════════════════════════════════╗
║       CHECKLIST PRE-INDEXACIÓN - VERIFICACIÓN RÁPIDA      ║
╚═══════════════════════════════════════════════════════════╝
""")

# Cargar variables de entorno
load_dotenv()

# Lista de verificaciones
checks = []

# 1. API Key
print("1️⃣  Verificando API Key de Google...")
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    print(f"   ✅ API Key presente (longitud: {len(api_key)} caracteres)")
    checks.append(True)
else:
    print("   ❌ API Key NO encontrada")
    print("      Configure con: $env:GOOGLE_API_KEY = 'tu-api-key'")
    checks.append(False)

# 2. Directorio de documentos
print("\n2️⃣  Verificando directorio documentos_srt/...")
if os.path.exists("documentos_srt"):
    srt_files = [f for f in os.listdir("documentos_srt") if f.endswith(".srt")]
    print(f"   ✅ Directorio existe con {len(srt_files)} archivos .srt")
    checks.append(True)
else:
    print("   ❌ Directorio documentos_srt/ NO existe")
    checks.append(False)

# 3. Espacio en disco
print("\n3️⃣  Verificando espacio en disco...")
try:
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    if free_gb > 1:
        print(f"   ✅ Espacio libre: {free_gb:.2f} GB")
        checks.append(True)
    else:
        print(f"   ⚠️ Espacio libre bajo: {free_gb:.2f} GB (recomendado >1GB)")
        checks.append(False)
except:
    print("   ⚠️ No se pudo verificar espacio en disco")
    checks.append(True)  # No crítico

# 4. Dependencias
print("\n4️⃣  Verificando dependencias Python...")
missing = []
try:
    import langchain
    import langchain_google_genai
    import langchain_community
    import faiss
    print("   ✅ Todas las dependencias críticas instaladas")
    checks.append(True)
except ImportError as e:
    print(f"   ❌ Falta instalar: {e.name}")
    missing.append(e.name)
    checks.append(False)

# 5. Script de indexación
print("\n5️⃣  Verificando script reiniciar_indice.py...")
if os.path.exists("reiniciar_indice.py"):
    with open("reiniciar_indice.py", "r", encoding="utf-8") as f:
        content = f.read()
        if "CHUNK_SIZE = 300" in content and "PAUSE_EVERY" in content:
            print("   ✅ Script optimizado y protegido encontrado")
            checks.append(True)
        else:
            print("   ⚠️ Script existe pero puede no tener las últimas protecciones")
            checks.append(False)
else:
    print("   ❌ Script reiniciar_indice.py NO encontrado")
    checks.append(False)

# 6. Backup previo
print("\n6️⃣  Verificando índice actual...")
if os.path.exists("faiss_index"):
    if os.path.exists("faiss_index/index.faiss"):
        size = os.path.getsize("faiss_index/index.faiss") / (1024**2)
        print(f"   ✅ Índice actual existe ({size:.2f} MB)")
        print("      Se creará backup automáticamente")
        checks.append(True)
    else:
        print("   ⚠️ Directorio existe pero índice corrupto")
        checks.append(False)
else:
    print("   ℹ️ No hay índice previo (primera indexación)")
    checks.append(True)

# 7. Test de conexión Google
print("\n7️⃣  Probando conexión con Google Generative AI...")
if api_key:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        test_result = embeddings.embed_query("test")
        if test_result and len(test_result) > 0:
            print(f"   ✅ Conexión exitosa (dimensión: {len(test_result)})")
            checks.append(True)
        else:
            print("   ❌ Conexión falló (respuesta vacía)")
            checks.append(False)
    except Exception as e:
        print(f"   ❌ Error de conexión: {str(e)[:80]}")
        checks.append(False)
else:
    print("   ⏭️ Saltado (sin API key)")
    checks.append(False)

# Resumen
print("\n" + "="*60)
print("📊 RESUMEN")
print("="*60)

total = len(checks)
passed = sum(checks)
failed = total - passed

print(f"✅ Verificaciones exitosas: {passed}/{total}")
print(f"❌ Verificaciones fallidas: {failed}/{total}")

if all(checks):
    print("\n" + "🎉"*20)
    print("\n✅ ¡TODO LISTO PARA LA INDEXACIÓN!")
    print("\nPuedes ejecutar:")
    print("   python reiniciar_indice.py")
    print("\n" + "🎉"*20)
    sys.exit(0)
else:
    print("\n" + "⚠️"*20)
    print("\n⚠️ CORRIGE LOS ERRORES ANTES DE CONTINUAR")
    
    if not api_key:
        print("\nPRIORIDAD 1: Configura tu API Key:")
        print("   $env:GOOGLE_API_KEY = 'tu-api-key'")
    
    if missing:
        print("\nPRIORIDAD 2: Instala dependencias:")
        print("   pip install -r requirements.txt")
    
    print("\n" + "⚠️"*20)
    sys.exit(1)

"""
Script de prueba para verificar si Google Sheets Logger está operativo
"""

import os
from datetime import datetime

print("=" * 60)
print("TEST: Google Sheets Logger - Estado de Operatividad")
print("=" * 60)

# 1. Verificar archivo de credenciales
print("\n[1] Verificando archivo de credenciales...")
credentials_file = "google_credentials.json"
if os.path.exists(credentials_file):
    print(f"   ✅ Archivo '{credentials_file}' existe")
    file_size = os.path.getsize(credentials_file)
    print(f"   📊 Tamaño: {file_size} bytes")
else:
    print(f"   ❌ Archivo '{credentials_file}' NO existe")

# 2. Verificar dependencias
print("\n[2] Verificando dependencias...")
try:
    import gspread
    print(f"   ✅ gspread instalado (versión: {gspread.__version__})")
except ImportError as e:
    print(f"   ❌ gspread NO instalado: {e}")

try:
    from oauth2client.service_account import ServiceAccountCredentials
    print("   ✅ oauth2client instalado")
except ImportError as e:
    print(f"   ❌ oauth2client NO instalado: {e}")

# 3. Verificar import del módulo
print("\n[3] Verificando módulo google_sheets_logger.py...")
try:
    from google_sheets_logger import GoogleSheetsLogger, create_sheets_logger
    print("   ✅ Módulo importado correctamente")
except ImportError as e:
    print(f"   ❌ Error al importar módulo: {e}")
    exit(1)

# 4. Intentar crear el logger
print("\n[4] Intentando crear instancia del logger...")
try:
    logger = create_sheets_logger()
    if logger:
        print("   ✅ Logger creado exitosamente")
        print(f"   📋 Tipo: {type(logger)}")
    else:
        print("   ⚠️  Logger es None (posiblemente no configurado)")
except Exception as e:
    print(f"   ❌ Error al crear logger: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. Intentar conectar a Google Sheets (solo si logger existe)
if logger:
    print("\n[5] Intentando conectar a Google Sheets...")
    try:
        # Intentar hacer una operación simple para verificar conexión
        test_data = {
            "interaction_id": "TEST_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            "user": "TEST_USER",
            "question": "Pregunta de prueba",
            "answer": [{"type": "normal", "content": "Respuesta de prueba"}],
            "device_info": {"device_type": "test", "browser": "test", "os": "test"},
            "location_info": {"city": "Test City", "country": "Test Country", "ip": "0.0.0.0"},
            "timing": {"total_time": 1.5},
            "success": True
        }
        
        print("   🔄 Enviando registro de prueba...")
        logger.log_interaction(**test_data)
        print("   ✅ Registro de prueba enviado exitosamente")
        print("   📊 Verifica en tu Google Sheet si apareció la fila de prueba")
        
    except Exception as e:
        print(f"   ❌ Error al conectar/enviar: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("FIN DEL TEST")
print("=" * 60)

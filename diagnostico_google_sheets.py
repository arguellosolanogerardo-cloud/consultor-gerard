"""
Script de diagnóstico completo para Google Sheets Logger
"""

import os
from datetime import datetime

print("=" * 70)
print("DIAGNÓSTICO COMPLETO: Google Sheets Logger")
print("=" * 70)

# 1. Verificar credenciales
print("\n[1] VERIFICACIÓN DE CREDENCIALES")
print("-" * 70)
credentials_file = "google_credentials.json"

if os.path.exists(credentials_file):
    print(f"✅ Archivo '{credentials_file}' EXISTE")
    file_size = os.path.getsize(credentials_file)
    print(f"   Tamaño: {file_size} bytes")
    
    # Intentar leer y parsear el JSON
    try:
        import json
        with open(credentials_file, 'r') as f:
            creds_data = json.load(f)
        
        print(f"✅ Archivo JSON VÁLIDO")
        print(f"   Type: {creds_data.get('type', 'N/A')}")
        print(f"   Project ID: {creds_data.get('project_id', 'N/A')}")
        print(f"   Client Email: {creds_data.get('client_email', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error parseando JSON: {e}")
else:
    print(f"❌ Archivo '{credentials_file}' NO EXISTE")
    print("\n⚠️  SOLUCIÓN:")
    print("   1. Sigue las instrucciones en GOOGLE_SHEETS_SETUP.md")
    print("   2. Descarga el archivo JSON de credenciales")
    print("   3. Guárdalo como 'google_credentials.json' en la raíz del proyecto")
    exit(1)

# 2. Verificar dependencias
print("\n[2] VERIFICACIÓN DE DEPENDENCIAS")
print("-" * 70)

try:
    import gspread
    print(f"✅ gspread INSTALADO (v{gspread.__version__})")
except ImportError:
    print("❌ gspread NO INSTALADO")
    print("   Instalar: pip install gspread")
    exit(1)

try:
    from oauth2client.service_account import ServiceAccountCredentials
    print("✅ oauth2client INSTALADO")
except ImportError:
    print("❌ oauth2client NO INSTALADO")
    print("   Instalar: pip install oauth2client")
    exit(1)

# 3. Intentar crear el logger
print("\n[3] CREACIÓN DEL LOGGER")
print("-" * 70)

try:
    from google_sheets_logger import GoogleSheetsLogger, create_sheets_logger
    print("✅ Módulo importado correctamente")
except ImportError as e:
    print(f"❌ Error importando módulo: {e}")
    exit(1)

# Crear logger con verbosidad
logger = GoogleSheetsLogger()

if logger.enabled:
    print(f"✅ Logger HABILITADO")
    print(f"   Spreadsheet: {logger.spreadsheet_name}")
    print(f"   Worksheet: {logger.worksheet_name}")
else:
    print(f"❌ Logger NO HABILITADO")
    print("\n⚠️  POSIBLES CAUSAS:")
    print("   1. La hoja de Google Sheets no existe")
    print("   2. La hoja no está compartida con el service account email")
    print("   3. Permisos insuficientes")
    print("\n📝 VERIFICA:")
    
    try:
        with open(credentials_file, 'r') as f:
            import json
            creds_data = json.load(f)
            service_email = creds_data.get('client_email', 'N/A')
            print(f"   Service Account Email: {service_email}")
            print(f"\n   1. Crea una hoja llamada: 'GERARD - Logs de Usuarios'")
            print(f"   2. Compártela con: {service_email}")
            print(f"   3. Dale permisos de EDITOR")
    except:
        pass
    
    exit(1)

# 4. Intentar obtener estadísticas
print("\n[4] VERIFICACIÓN DE CONEXIÓN")
print("-" * 70)

try:
    stats = logger.get_stats()
    print(f"✅ Conexión EXITOSA a Google Sheets")
    print(f"   Interacciones registradas: {stats.get('total_interactions', 0)}")
    print(f"   Usuarios únicos: {stats.get('unique_users', 0)}")
except Exception as e:
    print(f"❌ Error obteniendo estadísticas: {e}")

# 5. Intentar registrar una interacción de prueba
print("\n[5] TEST DE REGISTRO")
print("-" * 70)

test_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
test_data = {
    "interaction_id": test_id,
    "user": "USUARIO_PRUEBA",
    "question": "¿Esta es una pregunta de prueba?",
    "answer": "Esta es una respuesta de prueba en formato de texto simple sin JSON.",
    "device_info": {
        "device_type": "Desktop",
        "browser": "Chrome",
        "os": "Windows"
    },
    "location_info": {
        "city": "Ciudad de Prueba",
        "country": "País de Prueba",
        "ip": "192.168.1.1"
    },
    "timing": {
        "total_time": 2.5
    },
    "success": True
}

try:
    print(f"🔄 Enviando registro de prueba...")
    print(f"   ID: {test_id}")
    print(f"   Usuario: {test_data['user']}")
    print(f"   Pregunta: {test_data['question']}")
    
    logger.log_interaction(**test_data)
    
    print(f"\n✅ REGISTRO ENVIADO EXITOSAMENTE")
    print(f"\n📊 VERIFICA EN GOOGLE SHEETS:")
    print(f"   https://docs.google.com/spreadsheets/")
    print(f"   Busca la hoja: 'GERARD - Logs de Usuarios'")
    print(f"   Pestaña: 'Interacciones'")
    print(f"   Última fila debe contener ID: {test_id}")
    
except Exception as e:
    print(f"\n❌ ERROR AL REGISTRAR: {e}")
    import traceback
    print("\n📋 TRACEBACK COMPLETO:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("FIN DEL DIAGNÓSTICO")
print("=" * 70)

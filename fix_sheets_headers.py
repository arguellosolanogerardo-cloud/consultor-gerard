"""
Script para actualizar los headers de Google Sheets y hacer una prueba de registro.
"""

from google_sheets_logger import GoogleSheetsLogger
from datetime import datetime

def fix_headers():
    """Actualiza los headers de Google Sheets para eliminar Timestamp Unix."""
    print("🔧 Actualizando headers de Google Sheets...")
    
    try:
        # Inicializar el logger
        logger = GoogleSheetsLogger()
        
        if not logger.enabled:
            print("❌ Google Sheets Logger no está habilitado")
            return False
        
        # Actualizar headers
        logger._setup_headers()
        print("✅ Headers actualizados exitosamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error actualizando headers: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging():
    """Hace una prueba de registro en Google Sheets."""
    print("\n🧪 Realizando prueba de registro...")
    
    try:
        # Inicializar el logger
        logger = GoogleSheetsLogger()
        
        if not logger.enabled:
            print("❌ Google Sheets Logger no está habilitado")
            return False
        
        # Datos de prueba
        test_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        device_info = {
            "device_type": "Desktop",
            "browser": "Chrome",
            "os": "Windows"
        }
        
        location_info = {
            "city": "Ciudad de Prueba",
            "country": "País de Prueba",
            "ip": "192.168.1.1"
        }
        
        timing = {
            "total_time": 2.5
        }
        
        # Registrar interacción de prueba
        logger.log_interaction(
            interaction_id=test_id,
            user="USUARIO_PRUEBA",
            question="¿Esta es una pregunta de prueba?",
            answer="Esta es una respuesta de prueba en formato de texto simple sin JSON.",
            device_info=device_info,
            location_info=location_info,
            timing=timing,
            success=True
        )
        
        print(f"✅ Prueba registrada exitosamente con ID: {test_id}")
        print("\n📋 Datos registrados:")
        print(f"   - Usuario: USUARIO_PRUEBA")
        print(f"   - Dispositivo: {device_info['device_type']}")
        print(f"   - Navegador: {device_info['browser']}")
        print(f"   - OS: {device_info['os']}")
        print(f"   - Ciudad: {location_info['city']}")
        print(f"   - País: {location_info['country']}")
        print(f"   - IP: {location_info['ip']}")
        print(f"   - Tiempo: {timing['total_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de registro: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 CORRECCIÓN DE HEADERS Y PRUEBA DE GOOGLE SHEETS")
    print("=" * 60)
    
    # Paso 1: Actualizar headers
    headers_ok = fix_headers()
    
    if headers_ok:
        # Paso 2: Hacer prueba de registro
        test_ok = test_logging()
        
        if test_ok:
            print("\n" + "=" * 60)
            print("✅ TODO FUNCIONÓ CORRECTAMENTE")
            print("=" * 60)
            print("\n📊 Revisa tu Google Sheet 'GERARD - Logs de Usuarios'")
            print("   Deberías ver:")
            print("   1. Headers actualizados (14 columnas, sin Timestamp Unix)")
            print("   2. Un nuevo registro de prueba con todos los datos correctos")
        else:
            print("\n❌ La prueba de registro falló")
    else:
        print("\n❌ No se pudieron actualizar los headers")

"""
Script para probar el registro con una respuesta en formato JSON real.
"""

from google_sheets_logger import GoogleSheetsLogger
from datetime import datetime
import json

def test_json_cleaning():
    """Prueba el registro con una respuesta JSON real como las que genera GERARD."""
    print("🧪 Probando registro con respuesta JSON real...")
    
    try:
        # Inicializar el logger
        logger = GoogleSheetsLogger()
        
        if not logger.enabled:
            print("❌ Google Sheets Logger no está habilitado")
            return False
        
        # Datos de prueba
        test_id = f"TEST_JSON_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simular respuesta JSON como la que genera GERARD
        json_response = """```json
[
  {
    "type": "normal",
    "content": "El amor es la esencia de todo lo creado. (Fuente: Meditacion_001.srt, Timestamp: 00:05:30 --> 00:05:45)"
  },
  {
    "type": "emphasis",
    "content": "Solo a través del amor podemos encontrar la verdadera paz. (Fuente: Meditacion_002.srt, Timestamp: 00:12:15 --> 00:12:30)"
  },
  {
    "type": "normal",
    "content": "La luz divina nos guía en cada momento de nuestra existencia. (Fuente: Meditacion_003.srt, Timestamp: 00:18:20 --> 00:18:40)"
  }
]
```"""
        
        # Importar la función de limpieza desde consultar_web.py
        import sys
        import re
        
        def clean_json_text(json_string: str) -> str:
            """Función de limpieza copiada de consultar_web.py"""
            try:
                # Remover backticks de markdown si existen
                json_string = re.sub(r'^```json\s*', '', json_string.strip())
                json_string = re.sub(r'\s*```$', '', json_string.strip())
                
                match = re.search(r'\[.*\]', json_string, re.DOTALL)
                if not match:
                    return json_string

                data = json.loads(match.group(0))
                # Concatenar todo el contenido de los items
                clean_text = " ".join([item.get("content", "") for item in data])
                return clean_text
            except Exception as ex:
                print(f"[DEBUG] ERROR limpiando JSON: {ex}")
                return json_string
        
        # Limpiar el JSON
        cleaned_answer = clean_json_text(json_response)
        
        print(f"\n📝 Respuesta original (JSON):")
        print(f"   {json_response[:100]}...")
        print(f"\n✨ Respuesta limpia (texto):")
        print(f"   {cleaned_answer[:150]}...")
        
        device_info = {
            "device_type": "Mobile",
            "browser": "Safari",
            "os": "iOS"
        }
        
        location_info = {
            "city": "Bogotá",
            "country": "Colombia",
            "ip": "181.129.45.67"
        }
        
        timing = {
            "total_time": 5.75
        }
        
        # Registrar interacción de prueba
        logger.log_interaction(
            interaction_id=test_id,
            user="PRUEBA_JSON",
            question="¿Qué es el amor según GERARD?",
            answer=cleaned_answer,  # Usar respuesta limpia
            device_info=device_info,
            location_info=location_info,
            timing=timing,
            success=True
        )
        
        print(f"\n✅ Prueba JSON registrada exitosamente con ID: {test_id}")
        print("\n📋 Verifica en Google Sheets que:")
        print("   1. La respuesta NO tiene ```json ni corchetes")
        print("   2. Solo aparece el texto limpio con las fuentes")
        print("   3. Dispositivo: Mobile")
        print("   4. Navegador: Safari")
        print("   5. OS: iOS")
        print("   6. Ciudad: Bogotá")
        print("   7. País: Colombia")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba JSON: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 PRUEBA DE LIMPIEZA DE JSON EN GOOGLE SHEETS")
    print("=" * 70)
    
    test_json_cleaning()
    
    print("\n" + "=" * 70)
    print("📊 Revisa tu Google Sheet ahora")
    print("=" * 70)

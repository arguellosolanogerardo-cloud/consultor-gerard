"""
Script para convertir google_credentials.json al formato correcto para Streamlit Secrets.
"""

import json
import os

def convert_credentials_to_toml():
    """Convierte las credenciales JSON a formato TOML para Streamlit Cloud."""
    
    credentials_file = "google_credentials.json"
    
    if not os.path.exists(credentials_file):
        print(f"❌ Archivo no encontrado: {credentials_file}")
        return
    
    try:
        # Leer el archivo JSON
        with open(credentials_file, 'r', encoding='utf-8') as f:
            creds = json.load(f)
        
        # Generar formato TOML
        toml_output = f"""
# Copia EXACTAMENTE este contenido en Streamlit Cloud → Settings → Secrets

# Google API Key (reemplaza con tu clave real)
GOOGLE_API_KEY = "TU_GOOGLE_API_KEY_AQUI"

# Google Sheets Service Account
[gcp_service_account]
type = "{creds.get('type', '')}"
project_id = "{creds.get('project_id', '')}"
private_key_id = "{creds.get('private_key_id', '')}"
private_key = "{creds.get('private_key', '').replace(chr(10), '\\n')}"
client_email = "{creds.get('client_email', '')}"
client_id = "{creds.get('client_id', '')}"
auth_uri = "{creds.get('auth_uri', '')}"
token_uri = "{creds.get('token_uri', '')}"
auth_provider_x509_cert_url = "{creds.get('auth_provider_x509_cert_url', '')}"
client_x509_cert_url = "{creds.get('client_x509_cert_url', '')}"
universe_domain = "{creds.get('universe_domain', 'googleapis.com')}"
"""
        
        # Guardar en archivo
        output_file = "streamlit_secrets_format.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(toml_output)
        
        print("=" * 70)
        print("✅ Conversión exitosa!")
        print("=" * 70)
        print(f"\n📄 El formato TOML se ha guardado en: {output_file}")
        print("\n📋 INSTRUCCIONES:")
        print("1. Abre el archivo: streamlit_secrets_format.txt")
        print("2. Copia TODO el contenido")
        print("3. Ve a Streamlit Cloud → Tu App → Settings → Secrets")
        print("4. Pega el contenido completo")
        print("5. Reemplaza 'TU_GOOGLE_API_KEY_AQUI' con tu clave real")
        print("6. Haz clic en 'Save'")
        print("\n⚠️  IMPORTANTE:")
        print("   - NO modifiques los saltos de línea (\\n) en private_key")
        print("   - NO agregues comillas adicionales")
        print("   - NO elimines las comillas existentes")
        print("\n🔍 Verifica que estos valores sean correctos:")
        print(f"   - project_id: {creds.get('project_id', 'NO ENCONTRADO')}")
        print(f"   - client_email: {creds.get('client_email', 'NO ENCONTRADO')}")
        print("\n" + "=" * 70)
        
        # También mostrar en pantalla (primeros caracteres)
        print("\n📝 Vista previa (primeros 500 caracteres):")
        print("-" * 70)
        print(toml_output[:500])
        print("...")
        print("-" * 70)
        
    except Exception as e:
        print(f"❌ Error al convertir credenciales: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_credentials_to_toml()

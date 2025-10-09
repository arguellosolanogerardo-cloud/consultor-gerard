"""Store Google API key in the OS keyring for this project.

Usage (PowerShell):
$env:GOOGLE_API_KEY = '<YOUR_KEY>'
python .\scripts\store_key_keyring.py

Or run interactively and paste the key when prompted.
"""
import keyring
import os

key = os.environ.get('GOOGLE_API_KEY')
if not key:
    key = input('Introduce la API key de Google (no se mostrará): ').strip()

if key:
    keyring.set_password('consultor-gerard', 'google_api_key', key)
    print('Clave guardada en el keyring del sistema con servicio "consultor-gerard" y nombre "google_api_key"')
else:
    print('No se recibió clave. Nada guardado.')

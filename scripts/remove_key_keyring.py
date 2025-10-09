"""Remove the stored Google API key from the OS keyring.

Usage:
python .\scripts\remove_key_keyring.py
"""
import keyring

try:
    keyring.delete_password('consultor-gerard', 'google_api_key')
    print('Clave eliminada del keyring (si existía).')
except Exception as e:
    print('No se pudo eliminar la clave del keyring:', e)

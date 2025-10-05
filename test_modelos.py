"""
Este script se conecta a la API de Google Generative AI para obtener y mostrar
una lista de todos los modelos de lenguaje disponibles para tu clave de API.

Se enfoca en los modelos que soportan el método 'generateContent', que son los
que se utilizan para tareas de chat y generación de texto como en tu aplicación.

Uso:
1. Asegúrate de tener tu GOOGLE_API_KEY en el archivo .env.
2. Ejecuta este script desde tu terminal con el entorno virtual activado:
   python test_modelos.py
   o
   .\.venv\Scripts\python.exe test_modelos.py
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar la clave de API desde el archivo .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: No se encontró la GOOGLE_API_KEY en el archivo .env")
    exit()

# Configurar la API
genai.configure(api_key=api_key)

print("Buscando modelos disponibles para tu clave de API...")
print("-" * 50)

try:
    # Obtener la lista de todos los modelos
    models = genai.list_models()

    # Filtrar solo los modelos que se pueden usar para generar contenido (chat)
    generative_models = [m for m in models if 'generateContent' in m.supported_generation_methods]

    if not generative_models:
        print("No se encontraron modelos compatibles con 'generateContent' para tu clave de API.")
        print("Esto podría indicar un problema de permisos o configuración en tu proyecto de Google Cloud.")
    else:
        print("Modelos de Chat/Generación Encontrados:")
        print("-" * 50)
        print(f"{'Nombre del Modelo':<35} | {'Descripción'}")
        print(f"{'-'*35} | {'-'*40}")
        for model in generative_models:
            # El nombre que necesitamos para el código es `model.name`
            print(f"{model.name:<35} | {model.description}")
        print("-" * 50)
        print("\nRecomendación:")
        print("Copia el nombre exacto de la primera columna (ej: 'models/gemini-1.5-pro-latest') y pégalo en tus archivos 'consultar_terminal.py' y 'consultar_web.py'.")

except Exception as e:
    print(f"Ocurrió un error al intentar contactar a la API de Google: {e}")
    print("Verifica que tu clave de API sea correcta y que tengas el servicio 'Generative Language API' habilitado en tu proyecto de Google Cloud.")


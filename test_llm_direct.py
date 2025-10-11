#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test directo del LLM para diagnosticar por qué devuelve []
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

llm = GoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.4,
    top_p=0.90,
    top_k=25
)

# Simular lo que recibe el LLM (contexto fragmentado)
contexto_fragmentado = """
Fuente: Una reflexión de cómo hemos llegado aquí ala tierra [PPe2WYM8CP8].es.srt
Contenido:
327
00:13:01,760 --> 00:13:07,440
linajes, linaje ra, linaje linaje vix.

328
00:13:05,079 --> 00:13:10,639
Okay. Son cuatro linajes. Y el otro

329
00:13:07,440 --> 00:13:13,160
linaje eh Crick o Tri
"""

prompt = f"""Eres GERARD, un asistente espiritual. Tu misión es responder con precisión quirúrgica.

FORMATO DE SALIDA OBLIGATORIO (JSON):

[
  {{"type": "normal", "content": "Texto con cita (Fuente: archivo.srt, Timestamp: HH:MM:SS)"}}
]

Contexto disponible:
{contexto_fragmentado}

Consulta del usuario: INFORMACION SOBRE LINAJE RA, BIS, TRICK, JAC

Responde en formato JSON con citas obligatorias.
"""

print("="*70)
print("ENVIANDO AL LLM:")
print("="*70)
print(prompt[:500] + "...\n")

print("="*70)
print("RESPUESTA DEL LLM:")
print("="*70)
respuesta = llm.invoke(prompt)
print(respuesta)
print("\n" + "="*70)
print(f"Tipo: {type(respuesta)}")
print(f"Longitud: {len(respuesta) if isinstance(respuesta, str) else 'N/A'}")

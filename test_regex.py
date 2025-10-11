import re

respuesta_llm = """```json
[
  {
    "type": "normal",
    "content": "El contexto confirma..."
  }
]
```"""

print("Respuesta LLM:")
print(respuesta_llm)
print("\n" + "="*70)

# Test 1: Regex actual
match = re.search(r'\[.*\]', respuesta_llm, re.DOTALL)
print(f"Match encontrado: {match is not None}")
if match:
    print(f"JSON extraído:\n{match.group(0)}")
else:
    print("❌ NO SE ENCONTRÓ MATCH")

# Test 2: Limpiando markdown primero
cleaned = re.sub(r'```json\s*|\s*```', '', respuesta_llm)
print("\n" + "="*70)
print("Después de limpiar markdown:")
print(cleaned)

match2 = re.search(r'\[.*\]', cleaned, re.DOTALL)
print(f"\nMatch en texto limpio: {match2 is not None}")

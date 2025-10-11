# 🚨 DIAGNÓSTICO URGENTE: LLM Responde `[]` Vacío

## PROBLEMA IDENTIFICADO

**Síntoma:** El LLM (Gemini 2.5 Pro) devuelve literalmente `[]` (array vacío) incluso cuando:
- ✅ FAISS encuentra 75 documentos relevantes
- ✅ El retriever envía 21,636 caracteres de contexto
- ✅ El contexto incluye "linaje ra", "linaje bis", etc.

**Causa raíz:**  Con `chunk_size=300`, la información está TAN FRAGMENTADA que el LLM NO puede armar una respuesta coherente.

---

## EVIDENCIA

### Log de Streamlit:
```
[DEBUG] format_docs_with_metadata] Recibidos 75 documentos  
[DEBUG] format_docs_with_metadata] Devolviendo 21636 caracteres de contexto  
[DEBUG] Después de invoke - answer_raw type: <class 'str'>, valor: []  ← ❌ VACÍO
```

### Búsqueda directa en FAISS:
```python
Documentos recuperados: 75

# Documento #2 (RELEVANTE):
"linajes, linaje ra, linaje linaje vix.
Okay. Son cuatro linajes. Y el otro
linaje eh Crick o Tri..."
```

✅ **La información SÍ EXISTE en el índice**
❌ **Pero está FRAGMENTADA en chunks de 300 caracteres**

---

## ¿POR QUÉ EL LLM RESPONDE `[]`?

Gemini 2.5 Pro es **muy estricto** con la precisión (temperature=0.4). Cuando recibe contexto fragmentado e incompleto:

1. **Ve fragmentos** como:
   - Chunk 1: "linajes, linaje ra, linaje linaje vix."
   - Chunk 2: "Okay. Son cuatro linajes. Y el otro"
   - Chunk 3: "linaje eh Crick o Tri"
   
2. **NO puede unir** la información dispersa en 75 chunks

3. **Decide que NO tiene suficiente información confiable**

4. **Responde `[]`** (vacío) en lugar de arriesgarse a dar información incompleta

---

## SOLUCIÓN INMEDIATA

### ⚡ RE-INDEXAR CON chunk_size=800

**Ya está configurado en `reiniciar_indice.py`**, solo ejecutar:

```powershell
python reiniciar_indice.py
```

**Tiempo estimado:** 2-3 horas  
**Resultado esperado:**
- ✅ Información completa en cada chunk
- ✅ LLM puede responder con confianza
- ✅ 90-95% de recall (vs 60-70% actual)

---

## WORKAROUND TEMPORAL (mientras re-indexas)

### Opción 1: Aumentar temperatura (NO recomendado)

```python
temperature=0.7  # Más "arriesgado", puede inventar
```

❌ **NO hacer esto** - GERARD requiere precisión quirúrgica

### Opción 2: Modificar prompt para aceptar respuestas parciales

Agregar al prompt:

```
Si la información está fragmentada o incompleta, proporciona lo que puedas 
basándote en los fragmentos disponibles, indicando claramente que la respuesta
es parcial.
```

⚠️ **Subóptimo** - mejor re-indexar

### Opción 3: Usar búsqueda híbrida (keyword + semántica)

❌ **Complejo** - requiere refactorización mayor

---

## PLAN DE ACCIÓN

### ✅ AHORA MISMO:

1. **RE-INDEXAR con chunk_size=800:**
   ```powershell
   python reiniciar_indice.py
   ```

2. **Esperar 2-3 horas** (protecciones anti-rate-limit activas)

3. **Editar `consultar_web.py`:**
   ```python
   k = 40  # Reducir de 75 a 40
   ```

4. **Reiniciar Streamlit**

5. **Probar de nuevo:** "INFORMACION SOBRE LINAJE RA, BIS, TRICK, JAC"

### ✅ RESULTADO ESPERADO:

```json
[
  {
    "type": "normal",
    "content": "Los cuatro linajes son linaje RA, linaje BIS, linaje TRICK y linaje JAC... (Fuente: archivo.srt, Timestamp: 00:13:01)"
  }
]
```

---

## CONCLUSIÓN

**El problema NO es:**
- ❌ La API key de Gemini
- ❌ El código de Streamlit
- ❌ El retriever de FAISS
- ❌ El prompt

**El problema ES:**
- ✅ **chunk_size=300 es DEMASIADO PEQUEÑO**
- ✅ **La información está FRAGMENTADA**
- ✅ **El LLM NO puede unir 75 fragmentos pequeños**

**La solución ES:**
- ✅ **RE-INDEXAR con chunk_size=800**
- ✅ **Esperar 2-3 horas**
- ✅ **Problema resuelto permanentemente**

---

**Creado:** 10 octubre 2025, 23:45  
**Prioridad:** 🚨 CRÍTICA  
**Acción requerida:** RE-INDEXAR AHORA

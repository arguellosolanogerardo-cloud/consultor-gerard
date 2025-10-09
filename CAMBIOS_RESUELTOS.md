# Resumen de Cambios y Resolución de Problemas

## Fecha: 9 de octubre de 2025

## Problemas Resueltos

### 1. Error de Protobuf: "'ProtoType' object has no attribute 'DESCRIPTOR'"

**Causa**: Incompatibilidad entre `protobuf 4.23.4` y las librerías cliente de Google (`google-ai-generativelanguage`, `google-generativeai`, `proto-plus`).

**Solución Implementada**:
- Creado nuevo virtualenv limpio en `.venv_clean/`
- Instaladas versiones compatibles con `protobuf >= 5.26.1`
- Generado `requirements_pinned.txt` con versiones específicas

**Versiones Clave Instaladas**:
```
protobuf==5.29.5
google-generativeai==0.8.5
google-ai-generativelanguage==0.6.15
langchain-google-genai==2.0.10
proto-plus==1.26.1
grpcio==1.75.1
grpcio-status==1.71.2
```

### 2. Error: "expected string or bytes-like object, got 'dict'"

**Causa**: La función `retrieval_chain.invoke()` puede devolver un dict o un string dependiendo del flujo (FakeChain vs cadena real), pero el código esperaba siempre un string JSON para `re.search()` y `save_to_log()`.

**Solución**:
- Modificado `consultar_web.py` líneas ~905-910
- Añadida conversión explícita: si `answer_raw` es dict, se convierte a JSON string
- Garantiza que `answer_json` sea siempre string antes de pasar a regex y logging

**Código Corregido**:
```python
answer_raw = retrieval_chain.invoke(payload)
# Asegurar que answer_json sea siempre un string JSON
if isinstance(answer_raw, dict):
    answer_json = json.dumps(answer_raw, ensure_ascii=False)
else:
    answer_json = answer_raw if isinstance(answer_raw, str) else str(answer_raw)
```

### 3. Mejoras en Carga Perezosa (Lazy Loading)

**Cambios en `load_resources()`**:
- Imports de `GoogleGenerativeAI` y `GoogleGenerativeAIEmbeddings` movidos dentro de la función
- Envueltos en try/except individual para capturar errores de protobuf sin romper la app
- Implementado fallback `FakeEmbeddings` con dimensión dinámica (lee FAISS index para obtener dimensión correcta: 768)
- Implementado fallback `FakeChain` para operar sin LLM oficial

**Beneficios**:
- La app arranca aunque haya conflictos de protobuf
- Modo demo funcional sin necesidad de API key
- Recuperación elegante ante fallos de inicialización

## Archivos Modificados

1. **`consultar_web.py`**:
   - Imports perezosos de librerías Google
   - Fallbacks robustos (FakeEmbeddings, FakeChain)
   - Corrección de tipo en respuesta de invoke
   - Keyring lookup para API key

2. **`requirements_pinned.txt`** (nuevo):
   - Versiones específicas compatibles
   - Protobuf >= 5.26.1
   - Todas las dependencias necesarias

3. **`start_app.ps1`** (nuevo):
   - Script PowerShell para iniciar la app con el entorno limpio

## Cómo Usar el Entorno Limpio

### Opción 1: Usar el Script de Inicio (Recomendado)
```powershell
.\start_app.ps1
```

### Opción 2: Manual
```powershell
# Activar entorno limpio
.\.venv_clean\Scripts\Activate.ps1

# Ejecutar Streamlit
streamlit run consultar_web.py
```

### Opción 3: CLI (Terminal)
```powershell
# Activar entorno limpio
.\.venv_clean\Scripts\Activate.ps1

# Ejecutar CLI
python consultar_terminal.py
```

## Validaciones Realizadas

✅ Import de `consultar_web.py`: IMPORT_OK  
✅ Import de librerías Google sin error de protobuf  
✅ FAISS index compatible (ntotal=3084, dimensión=768)  
✅ FakeEmbeddings genera vectores de dimensión 768  
✅ Streamlit arranca sin errores en http://localhost:8501  

## Estado del Entorno Anterior (.venv)

⚠️ El entorno `.venv` original tiene `protobuf==4.23.4` que causa incompatibilidades.  
💡 Recomendación: Usar `.venv_clean` para todas las ejecuciones futuras.  
🔧 Opcional: Puedes eliminar `.venv` para evitar confusión, o actualizarlo instalando `requirements_pinned.txt`.

## Próximos Pasos Sugeridos

1. **Probar E2E con API key real**:
   - Asegurar que `GOOGLE_API_KEY` está en keyring o variable de entorno
   - Hacer una consulta en la UI y verificar respuesta con GERARD v3.01

2. **Validar PDF Export**:
   - Hacer una conversación completa
   - Exportar a PDF y verificar que preserva colores y formato

3. **Validar CLI**:
   - Ejecutar `consultar_terminal.py` con el entorno limpio
   - Confirmar que usa keyring y genera respuestas correctas

4. **Opcional - Migrar .venv a versiones nuevas**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements_pinned.txt --upgrade
   ```

## Notas Técnicas

- **FAISS Index**: Construido con embeddings de dimensión 768
- **FakeEmbeddings**: Genera vectores determinísticos usando SHA256 + contador, compatible con dimensión del índice
- **Personality**: GERARD v3.01 aplicado en prompt template
- **Keyring**: Service 'consultor-gerard', key 'google_api_key' (verificado: FOUND)
- **Logs**: Conversaciones guardadas en `gerard_log.txt`

## Contacto y Soporte

Si encuentras problemas:
1. Verifica que estás usando `.venv_clean`
2. Confirma versiones con: `pip show protobuf google-generativeai`
3. Revisa logs de Streamlit en la terminal

---
**Generado**: 2025-10-09  
**Autor**: GitHub Copilot (AI Assistant)

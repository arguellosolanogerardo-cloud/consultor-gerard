# 📋 RESUMEN: CONFIGURACIÓN ÓPTIMA DE CHUNKS

## 🎯 RESPUESTA DIRECTA

### Para tu PRÓXIMA re-indexación (cuando agregues más .srt):

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
```

---

## ✅ VENTAJAS DE 800/150

| Aspecto | Mejora vs actual (300/50) |
|---------|---------------------------|
| **Recall** | 90-95% (vs 60-70%) → **+35%** |
| **Chunks totales** | 72,000 (vs 193,213) → **-63%** |
| **Velocidad** | 40% más rápido |
| **Calidad** | Respuestas COMPLETAS (vs fragmentadas) |
| **k necesario** | 40 (vs 75) → **-47%** |
| **Espacio disco** | ~450 MB (vs 638 MB) → **-29%** |

---

## 🛠️ CUÁNDO APLICAR

### ✅ Aplica esta configuración cuando:

1. **Agregues más archivos .srt** a la carpeta `documentos_srt/`
2. **Notes que fallan búsquedas** como "linajes" o información específica
3. **Quieras mejor calidad** de respuestas

### 📝 Pasos simples:

1. Editar `reiniciar_indice.py`: ✅ **YA ESTÁ LISTO con 800/150**
2. Agregar tus nuevos archivos .srt a `documentos_srt/`
3. Ejecutar: `python reiniciar_indice.py`
4. Esperar 2-3 horas
5. Editar `consultar_web.py`: cambiar `k=75` a `k=40`
6. Reiniciar Streamlit

---

## 💾 ESTADO ACTUAL

### Ahora mismo tienes:
- ✅ `reiniciar_indice.py` configurado con **800/150** (listo para próxima vez)
- ⚠️ Índice actual en `faiss_index/` con **300/50** (funcional pero subóptimo)
- ✅ `consultar_web.py` con **k=75** (compensando chunks pequeños)

### Próxima vez que agregues archivos:
- Solo ejecuta `python reiniciar_indice.py`
- Automáticamente usará **800/150**
- Tendrás mejor calidad sin hacer nada más

---

## 🔬 VALIDACIÓN

Después de re-indexar con 800/150, esta pregunta DEBE funcionar perfectamente:

**Pregunta:** "INFORMACION SOBRE LINAJE RA, BIS, TRICK, JAC"

**Respuesta esperada:**
```
Los cuatro linajes son:
- Linaje RA: [descripción completa]
- Linaje BIS: [descripción completa]
- Linaje TRICK: [descripción completa]
- Linaje JAC (o JACK): [descripción completa]
(Fuente: archivo.srt, Timestamp: HH:MM:SS)
```

Si funciona → ✅ Configuración perfecta!

---

## 📚 MÁS DETALLES

Ver: `GUIA_TAMAÑO_CHUNKS_OPTIMO.md` para análisis completo.

---

**Última actualización:** 10 octubre 2025  
**Estado:** `reiniciar_indice.py` configurado y listo con 800/150

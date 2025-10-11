# 🎨 CONFIGURACIÓN DE TEMA OSCURO PERMANENTE

**Fecha**: 11 de octubre de 2025  
**Archivo**: `.streamlit/config.toml`

---

## ✅ **TEMA OSCURO FORZADO**

La aplicación **SIEMPRE** se mostrará con fondo negro, sin importar la configuración del navegador del usuario.

---

## 🎨 **CONFIGURACIÓN ACTUAL**

```toml
[theme]
base = "dark"                           # Tema base oscuro
primaryColor = "#FF4B4B"                # Rojo Streamlit (botones, links)
backgroundColor = "#0E1117"             # Fondo principal (negro)
secondaryBackgroundColor = "#262730"    # Fondo secundario (gris oscuro)
textColor = "#FAFAFA"                   # Texto (blanco/casi blanco)
font = "sans serif"                     # Fuente
```

---

## 🎨 **PALETA DE COLORES**

| Elemento | Color | Hex | Uso |
|----------|-------|-----|-----|
| **Fondo principal** | Negro azulado | `#0E1117` | Fondo de la app |
| **Fondo secundario** | Gris oscuro | `#262730` | Sidebar, cards, containers |
| **Texto** | Blanco | `#FAFAFA` | Texto principal |
| **Primary** | Rojo Streamlit | `#FF4B4B` | Botones, links, énfasis |

---

## 🔧 **PERSONALIZAR COLORES (OPCIONAL)**

Si quieres cambiar los colores, edita `.streamlit/config.toml`:

### **Opción 1: Negro puro**
```toml
[theme]
base = "dark"
primaryColor = "#00D4FF"        # Azul cyan
backgroundColor = "#000000"     # Negro puro
secondaryBackgroundColor = "#1A1A1A"
textColor = "#FFFFFF"
font = "sans serif"
```

### **Opción 2: Azul oscuro espacial**
```toml
[theme]
base = "dark"
primaryColor = "#FFD700"        # Dorado
backgroundColor = "#0A0E27"     # Azul espacial oscuro
secondaryBackgroundColor = "#1A1F3A"
textColor = "#E8E8E8"
font = "sans serif"
```

### **Opción 3: Verde Matrix**
```toml
[theme]
base = "dark"
primaryColor = "#00FF41"        # Verde neón
backgroundColor = "#0D0208"     # Negro casi puro
secondaryBackgroundColor = "#1A1A1A"
textColor = "#00FF41"           # Verde Matrix
font = "monospace"
```

### **Opción 4: Morado cyberpunk**
```toml
[theme]
base = "dark"
primaryColor = "#FF00FF"        # Magenta
backgroundColor = "#1A0033"     # Morado muy oscuro
secondaryBackgroundColor = "#2D004D"
textColor = "#E0E0E0"
font = "sans serif"
```

---

## 📱 **COMPATIBILIDAD**

✅ **Desktop**: Chrome, Firefox, Safari, Edge  
✅ **Mobile**: Chrome Android, Safari iOS  
✅ **Tablets**: iPad, Android tablets  

**NOTA**: El tema configurado en `config.toml` **tiene prioridad** sobre la preferencia del navegador (`prefers-color-scheme`).

---

## 🚀 **APLICAR CAMBIOS**

### **Desarrollo local**:
```powershell
# Reiniciar Streamlit para aplicar cambios
Get-Process | Where-Object {$_.ProcessName -eq "streamlit"} | Stop-Process -Force
streamlit run consultar_web.py
```

### **Streamlit Cloud**:
1. Hacer commit de `.streamlit/config.toml`
2. Push a GitHub
3. Streamlit Cloud auto-deploy (2-5 min)

---

## 🎯 **CSS PERSONALIZADO ADICIONAL (OPCIONAL)**

Si quieres más control, puedes agregar CSS custom en `consultar_web.py`:

```python
# Al inicio de consultar_web.py
st.markdown("""
<style>
    /* Forzar fondo negro en todo */
    .stApp {
        background-color: #000000 !important;
    }
    
    /* Sidebar negro */
    [data-testid="stSidebar"] {
        background-color: #1A1A1A !important;
    }
    
    /* Input fields con fondo oscuro */
    .stTextInput > div > div > input {
        background-color: #262730 !important;
        color: #FAFAFA !important;
    }
    
    /* Botones personalizados */
    .stButton > button {
        background-color: #FF4B4B !important;
        color: white !important;
        border: none !important;
    }
    
    /* Headers con color personalizado */
    h1, h2, h3 {
        color: #FF4B4B !important;
    }
</style>
""", unsafe_allow_html=True)
```

---

## 📊 **COMPARACIÓN: ANTES vs DESPUÉS**

### **Antes** (dependiente del navegador):
```
Usuario con modo claro → App en blanco 🌞
Usuario con modo oscuro → App en negro 🌙
```

### **Después** (tema forzado):
```
Usuario con modo claro → App en negro 🌙
Usuario con modo oscuro → App en negro 🌙
```

**Todos los usuarios ven el mismo tema oscuro** ✅

---

## 🔍 **VERIFICAR TEMA ACTUAL**

Agregar al sidebar de la app:

```python
# En consultar_web.py
with st.sidebar:
    st.info(f"🎨 Tema: OSCURO (forzado)")
```

---

## ⚠️ **NOTAS IMPORTANTES**

1. **Modo claro deshabilitado**: Los usuarios NO podrán cambiar a modo claro
2. **Accesibilidad**: Asegúrate de que el contraste sea suficiente (WCAG AA: 4.5:1)
3. **Imágenes**: Verifica que tus GIFs/imágenes se vean bien en fondo oscuro

---

## 🎨 **RECURSOS**

- **Streamlit Theming**: https://docs.streamlit.io/library/advanced-features/theming
- **Color Picker**: https://htmlcolorcodes.com/
- **Contrast Checker**: https://webaim.org/resources/contrastchecker/

---

## ✅ **ESTADO ACTUAL**

- ✅ Tema oscuro configurado en `.streamlit/config.toml`
- ✅ Fondo negro permanente (`#0E1117`)
- ✅ Colores Streamlit estándar (rojo `#FF4B4B`)
- ⏳ Pendiente: commit y push a GitHub

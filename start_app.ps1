# Script de inicio para GERARD - Consultor Web
# Usa el entorno virtual limpio (.venv_clean) con protobuf 5.x compatible

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "  GERARD - Consultor Web" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Iniciando aplicación Streamlit..." -ForegroundColor Yellow
Write-Host "Entorno: .venv_clean (protobuf 5.29.5)" -ForegroundColor Gray
Write-Host ""

# Activar entorno limpio y ejecutar Streamlit
& E:\proyecto-gemini\.venv_clean\Scripts\python.exe -m streamlit run E:\proyecto-gemini\consultar_web.py

Write-Host ""
Write-Host "Aplicación detenida." -ForegroundColor Yellow

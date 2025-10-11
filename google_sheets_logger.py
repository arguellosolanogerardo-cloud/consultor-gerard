"""
Google Sheets Logger para GERARD

Este módulo envía automáticamente cada interacción a una hoja de Google Sheets
para que puedas ver todos los logs de usuarios en tiempo real desde cualquier lugar.

Características:
- Registro automático en Google Sheets
- Columnas: Fecha/Hora, Usuario, Pregunta, Respuesta, Dispositivo, Navegador, OS, Ciudad, País, IP, Tiempo
- Acceso desde cualquier dispositivo
- Actualización en tiempo real
- Sin límites de almacenamiento (hasta 10M celdas)

Configuración:
1. Crear un proyecto en Google Cloud Console
2. Habilitar Google Sheets API
3. Crear credenciales (Service Account)
4. Descargar archivo JSON de credenciales
5. Compartir la hoja de Google Sheets con el email del service account
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from typing import Dict, Optional
import json
import os


class GoogleSheetsLogger:
    """
    Logger que envía interacciones a Google Sheets en tiempo real.
    """
    
    def __init__(
        self,
        credentials_file: str = "google_credentials.json",
        spreadsheet_name: str = "GERARD - Logs de Usuarios",
        worksheet_name: str = "Interacciones"
    ):
        """
        Inicializa el logger de Google Sheets.
        
        Args:
            credentials_file: Ruta al archivo JSON de credenciales
            spreadsheet_name: Nombre de la hoja de cálculo
            worksheet_name: Nombre de la pestaña/worksheet
        """
        self.credentials_file = credentials_file
        self.spreadsheet_name = spreadsheet_name
        self.worksheet_name = worksheet_name
        self.client = None
        self.worksheet = None
        self.enabled = False
        
        # Intentar conectar
        self._connect()
    
    def _connect(self):
        """Conecta con Google Sheets."""
        try:
            # Verificar si existe el archivo de credenciales
            if not os.path.exists(self.credentials_file):
                print(f"⚠️  Google Sheets Logger: Archivo de credenciales no encontrado: {self.credentials_file}")
                print("   Para activar Google Sheets, sigue las instrucciones en GOOGLE_SHEETS_SETUP.md")
                return
            
            # Definir el scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Autenticar
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_file,
                scope
            )
            self.client = gspread.authorize(creds)
            
            # Abrir o crear la hoja de cálculo
            try:
                spreadsheet = self.client.open(self.spreadsheet_name)
            except gspread.SpreadsheetNotFound:
                print(f"⚠️  Hoja '{self.spreadsheet_name}' no encontrada. Créala y compártela con el service account.")
                return
            
            # Abrir o crear el worksheet
            try:
                self.worksheet = spreadsheet.worksheet(self.worksheet_name)
            except gspread.WorksheetNotFound:
                # Crear nuevo worksheet con encabezados
                self.worksheet = spreadsheet.add_worksheet(
                    title=self.worksheet_name,
                    rows=1000,
                    cols=15
                )
                self._setup_headers()
            
            self.enabled = True
            print(f"✅ Google Sheets Logger conectado exitosamente: {self.spreadsheet_name}")
            
        except Exception as e:
            print(f"⚠️  Error conectando con Google Sheets: {e}")
            print("   El logging continuará localmente sin Google Sheets")
    
    def _setup_headers(self):
        """Configura los encabezados de la hoja."""
        headers = [
            "ID",
            "Fecha/Hora",
            "Usuario",
            "Pregunta",
            "Respuesta (Resumen)",
            "Dispositivo",
            "Navegador",
            "Sistema Operativo",
            "Ciudad",
            "País",
            "IP",
            "Tiempo Respuesta (s)",
            "Estado",
            "Error",
            "Timestamp Unix"
        ]
        
        self.worksheet.update('A1:O1', [headers])
        
        # Formatear encabezados (negrita, fondo gris)
        self.worksheet.format('A1:O1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
        })
    
    def log_interaction(
        self,
        interaction_id: str,
        user: str,
        question: str,
        answer: str,
        device_info: Optional[Dict] = None,
        location_info: Optional[Dict] = None,
        timing: Optional[Dict] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Registra una interacción en Google Sheets.
        
        Args:
            interaction_id: ID único de la interacción
            user: Nombre del usuario
            question: Pregunta realizada
            answer: Respuesta generada
            device_info: Información del dispositivo
            location_info: Información de ubicación
            timing: Información de tiempos
            success: Si fue exitosa
            error: Mensaje de error si aplica
        """
        if not self.enabled:
            return
        
        try:
            # Preparar datos
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_unix = int(timestamp.timestamp())
            
            # Información del dispositivo
            device_type = "Desconocido"
            browser = "Desconocido"
            os_type = "Desconocido"
            
            if device_info:
                device_type = device_info.get("device_type", "Desconocido")
                browser = device_info.get("browser", "Desconocido")
                os_type = device_info.get("os", "Desconocido")
            
            # Información de ubicación
            city = "Desconocida"
            country = "Desconocido"
            ip = "No disponible"
            
            if location_info:
                city = location_info.get("city", "Desconocida")
                country = location_info.get("country", "Desconocido")
                ip = location_info.get("ip", "No disponible")
            
            # Tiempo de respuesta
            response_time = 0
            if timing:
                response_time = timing.get("total_time", 0)
            
            # Resumen de la respuesta (primeros 200 caracteres)
            answer_summary = answer[:200] + "..." if len(answer) > 200 else answer
            
            # Estado
            status = "✅ Exitoso" if success else "❌ Error"
            error_msg = error if error else ""
            
            # Crear fila
            row = [
                interaction_id,
                timestamp_str,
                user,
                question,
                answer_summary,
                device_type,
                browser,
                os_type,
                city,
                country,
                ip,
                f"{response_time:.2f}",
                status,
                error_msg,
                timestamp_unix
            ]
            
            # Agregar fila a la hoja
            self.worksheet.append_row(row)
            
            print(f"✅ Interacción registrada en Google Sheets: {user} - {question[:50]}...")
            
        except Exception as e:
            print(f"⚠️  Error registrando en Google Sheets: {e}")
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas de la hoja.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.enabled:
            return {}
        
        try:
            # Obtener todas las filas
            all_rows = self.worksheet.get_all_values()
            
            if len(all_rows) <= 1:  # Solo headers
                return {
                    "total_interactions": 0,
                    "unique_users": 0
                }
            
            # Contar (excluyendo header)
            data_rows = all_rows[1:]
            
            users = set()
            for row in data_rows:
                if len(row) > 2:
                    users.add(row[2])  # Columna de usuario
            
            return {
                "total_interactions": len(data_rows),
                "unique_users": len(users)
            }
            
        except Exception as e:
            print(f"⚠️  Error obteniendo estadísticas: {e}")
            return {}


# Función de ayuda para integración fácil
def create_sheets_logger() -> Optional[GoogleSheetsLogger]:
    """
    Crea y retorna un logger de Google Sheets.
    
    Returns:
        GoogleSheetsLogger o None si no está configurado
    """
    logger = GoogleSheetsLogger()
    return logger if logger.enabled else None

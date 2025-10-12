
"""
Google Sheets Logger para GERARD

Este modulo envia automaticamente cada interaccion a una hoja de Google Sheets
para que puedas ver todos los logs de usuarios en tiempo real desde cualquier lugar.

Caracteristicas:
- Registro automatico en Google Sheets
- Columnas: Fecha/Hora, Usuario, Pregunta, Respuesta, Dispositivo, Navegador, OS, Ciudad, Pais, IP, Tiempo
- Acceso desde cualquier dispositivo
- Actualizacion en tiempo real
- Sin limites de almacenamiento (hasta 10M celdas)

Configuracion:
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
    Logger que envia interacciones a Google Sheets en tiempo real.
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
            spreadsheet_name: Nombre de la hoja de calculo
            worksheet_name: Nombre de la pestana/worksheet
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
            # Definir el scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Intentar obtener credenciales desde Streamlit secrets primero
            creds = None
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
                    print("[INFO] Usando credenciales desde Streamlit secrets")
                    # Usar credenciales desde secrets
                    creds = ServiceAccountCredentials.from_json_keyfile_dict(
                        dict(st.secrets['gcp_service_account']),
                        scope
                    )
            except Exception as e:
                print(f"[INFO] No se pudieron cargar credenciales desde Streamlit secrets: {e}")
            
            # Si no hay credenciales desde secrets, intentar archivo local
            if creds is None:
                if not os.path.exists(self.credentials_file):
                    print(f"[!] Google Sheets Logger: Archivo de credenciales no encontrado: {self.credentials_file}")
                    print("    Para activar Google Sheets, sigue las instrucciones en GOOGLE_SHEETS_SETUP.md")
                    return
                
                print(f"[INFO] Usando credenciales desde archivo local: {self.credentials_file}")
                # Autenticar desde archivo
                creds = ServiceAccountCredentials.from_json_keyfile_name(
                    self.credentials_file,
                    scope
                )
            
            self.client = gspread.authorize(creds)
            
            # Abrir o crear la hoja de calculo
            try:
                spreadsheet = self.client.open(self.spreadsheet_name)
            except gspread.SpreadsheetNotFound:
                print(f"[!] Hoja '{self.spreadsheet_name}' no encontrada. Creala y compartela con el service account.")
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
            print(f"[OK] Google Sheets Logger conectado exitosamente: {self.spreadsheet_name}")
            
        except Exception as e:
            print(f"[!] Error conectando con Google Sheets: {e}")
            print("    El logging continuara localmente sin Google Sheets")
    
    def _setup_headers(self):
        """Configura los encabezados de la hoja."""
        headers = [
            "ID",
            "Fecha/Hora",
            "Usuario",
            "Pregunta",
            "Respuesta",
            "Dispositivo",
            "Navegador",
            "Sistema Operativo",
            "Ciudad",
            "Pais",
            "IP",
            "Tiempo Respuesta (s)",
            "Estado",
            "Error"
        ]
        
        self.worksheet.update('A1:N1', [headers])
        
        # Formatear encabezados (negrita, fondo gris)
        self.worksheet.format('A1:N1', {
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
        Registra una interaccion en Google Sheets.
        
        Args:
            interaction_id: ID unico de la interaccion
            user: Nombre del usuario
            question: Pregunta realizada
            answer: Respuesta generada
            device_info: Informacion del dispositivo
            location_info: Informacion de ubicacion
            timing: Informacion de tiempos
            success: Si fue exitosa
            error: Mensaje de error si aplica
        """
        if not self.enabled:
            return
        
        try:
            # Preparar datos
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Informacion del dispositivo
            device_type = "Desconocido"
            browser = "Desconocido"
            os_type = "Desconocido"
            
            if device_info:
                device_type = device_info.get("device_type", "Desconocido")
                browser = device_info.get("browser", "Desconocido")
                os_type = device_info.get("os", "Desconocido")
            
            # Informacion de ubicacion
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
            
            # Guardar la respuesta completa (sin límite de caracteres)
            answer_full = answer
            
            # Estado
            status = "[OK] Exitoso" if success else "[ERROR] Error"
            error_msg = error if error else ""
            
            # Crear fila
            row = [
                interaction_id,
                timestamp_str,
                user,
                question,
                answer_full,  # Respuesta completa
                device_type,
                browser,
                os_type,
                city,
                country,
                ip,
                f"{response_time:.2f}",
                status,
                error_msg
            ]
            
            # Agregar fila a la hoja
            self.worksheet.append_row(row)
            
            print(f"[OK] Interaccion registrada en Google Sheets: {user} - {question[:50]}...")
            
        except Exception as e:
            print(f"[!] Error registrando en Google Sheets: {e}")
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadisticas de la hoja.
        
        Returns:
            Diccionario con estadisticas
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
            print(f"[!] Error obteniendo estadisticas: {e}")
            return {}


# Funcion de ayuda para integracion facil
def create_sheets_logger() -> Optional[GoogleSheetsLogger]:
    """
    Crea y retorna un logger de Google Sheets.
    
    Returns:
        GoogleSheetsLogger o None si no esta configurado
    """
    logger = GoogleSheetsLogger()
    return logger if logger.enabled else None
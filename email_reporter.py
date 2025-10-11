"""
Sistema de Reportes por Email para GERARD

Este módulo genera y envía reportes diarios de las interacciones
con GERARD por correo electrónico.

Características:
- Resumen diario de interacciones
- Estadísticas de usuarios
- Top preguntas más frecuentes
- Información de dispositivos y ubicaciones
- Gráficos de uso (opcional)
- Envío automático por email

Configuración:
1. Editar EMAIL_CONFIG con tus datos
2. Para Gmail, activar "Aplicaciones menos seguras" o usar contraseña de aplicación
3. Ejecutar manualmente o programar con cron/Task Scheduler
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List
from collections import Counter
import os

# --- CONFIGURACIÓN DE EMAIL ---
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",  # Para Gmail
    "smtp_port": 587,  # Puerto TLS
    "sender_email": "tu_email@gmail.com",  # TU EMAIL AQUÍ
    "sender_password": "tu_contraseña_app",  # TU CONTRASEÑA DE APLICACIÓN AQUÍ
    "recipient_email": "destinatario@email.com",  # EMAIL DONDE RECIBIR REPORTES
    "subject_prefix": "[GERARD] Reporte Diario"
}


class EmailReporter:
    """
    Genera y envía reportes de interacciones por email.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Inicializa el generador de reportes.
        
        Args:
            log_dir: Directorio donde están los logs
        """
        self.log_dir = Path(log_dir)
        self.config = EMAIL_CONFIG
    
    def generate_daily_report(self, date: datetime = None) -> str:
        """
        Genera un reporte HTML para un día específico.
        
        Args:
            date: Fecha del reporte (default: ayer)
            
        Returns:
            String con el HTML del reporte
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        date_str = date.strftime("%Y%m%d")
        
        # Buscar archivos de log del día
        log_file = self.log_dir / f"interactions_{date_str}.log"
        json_file = self.log_dir / f"interactions_{date_str}.json"
        
        if not json_file.exists():
            return self._generate_no_data_report(date)
        
        # Cargar datos
        interactions = self._load_json_logs(json_file)
        
        if not interactions:
            return self._generate_no_data_report(date)
        
        # Generar estadísticas
        stats = self._calculate_statistics(interactions)
        
        # Generar HTML
        html = self._generate_html_report(date, stats, interactions)
        
        return html
    
    def _load_json_logs(self, json_file: Path) -> List[Dict]:
        """Carga las interacciones desde el archivo JSON."""
        interactions = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        interactions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error cargando logs: {e}")
        
        return interactions
    
    def _calculate_statistics(self, interactions: List[Dict]) -> Dict:
        """Calcula estadísticas de las interacciones."""
        stats = {
            "total_interactions": len(interactions),
            "unique_users": len(set(i.get("user_name", "Desconocido") for i in interactions)),
            "total_questions": len(interactions),
            "avg_response_time": 0,
            "top_users": [],
            "top_questions": [],
            "devices": Counter(),
            "locations": Counter(),
            "browsers": Counter(),
            "os_types": Counter(),
            "success_rate": 0
        }
        
        # Contadores
        user_counter = Counter()
        question_counter = Counter()
        response_times = []
        successful = 0
        
        for interaction in interactions:
            # Usuario
            user_name = interaction.get("user_name", "Desconocido")
            user_counter[user_name] += 1
            
            # Pregunta
            question = interaction.get("question", "")[:100]  # Primeros 100 chars
            question_counter[question] += 1
            
            # Tiempo de respuesta
            if "timing" in interaction and "total_time" in interaction["timing"]:
                response_times.append(interaction["timing"]["total_time"])
            
            # Success
            if interaction.get("success", False):
                successful += 1
            
            # Dispositivo
            device_info = interaction.get("device", {})
            device_type = device_info.get("device_type", "Desconocido")
            stats["devices"][device_type] += 1
            
            browser = device_info.get("browser", "Desconocido")
            stats["browsers"][browser] += 1
            
            os_type = device_info.get("os", "Desconocido")
            stats["os_types"][os_type] += 1
            
            # Ubicación
            location = interaction.get("location", {})
            city = location.get("city", "Desconocida")
            country = location.get("country", "Desconocido")
            location_str = f"{city}, {country}"
            stats["locations"][location_str] += 1
        
        # Promedios
        if response_times:
            stats["avg_response_time"] = sum(response_times) / len(response_times)
        
        stats["success_rate"] = (successful / len(interactions) * 100) if interactions else 0
        
        # Top 5
        stats["top_users"] = user_counter.most_common(5)
        stats["top_questions"] = question_counter.most_common(5)
        
        return stats
    
    def _generate_html_report(self, date: datetime, stats: Dict, interactions: List[Dict]) -> str:
        """Genera el HTML del reporte."""
        date_str = date.strftime("%d/%m/%Y")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f4f4f4;
                }}
                .header {{
                    background: linear-gradient(135deg, #8A2BE2 0%, #4B0082 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-card h3 {{
                    margin: 0 0 10px 0;
                    color: #8A2BE2;
                    font-size: 0.9em;
                    text-transform: uppercase;
                }}
                .stat-card .value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #333;
                }}
                .section {{
                    background: white;
                    padding: 25px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .section h2 {{
                    margin-top: 0;
                    color: #8A2BE2;
                    border-bottom: 2px solid #8A2BE2;
                    padding-bottom: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #8A2BE2;
                    color: white;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding: 20px;
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 GERARD</h1>
                <p>Reporte Diario de Interacciones</p>
                <p>{date_str}</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Interacciones</h3>
                    <div class="value">{stats['total_interactions']}</div>
                </div>
                <div class="stat-card">
                    <h3>Usuarios Únicos</h3>
                    <div class="value">{stats['unique_users']}</div>
                </div>
                <div class="stat-card">
                    <h3>Tiempo Promedio</h3>
                    <div class="value">{stats['avg_response_time']:.2f}s</div>
                </div>
                <div class="stat-card">
                    <h3>Tasa de Éxito</h3>
                    <div class="value">{stats['success_rate']:.1f}%</div>
                </div>
            </div>
            
            <div class="section">
                <h2>👥 Top 5 Usuarios Más Activos</h2>
                <table>
                    <tr>
                        <th>Usuario</th>
                        <th>Interacciones</th>
                    </tr>
        """
        
        for user, count in stats['top_users']:
            html += f"""
                    <tr>
                        <td>{user}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>💻 Dispositivos</h2>
                <table>
                    <tr>
                        <th>Tipo</th>
                        <th>Cantidad</th>
                    </tr>
        """
        
        for device, count in stats['devices'].most_common():
            html += f"""
                    <tr>
                        <td>{device}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>🌍 Ubicaciones</h2>
                <table>
                    <tr>
                        <th>Ciudad, País</th>
                        <th>Accesos</th>
                    </tr>
        """
        
        for location, count in stats['locations'].most_common():
            html += f"""
                    <tr>
                        <td>{location}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>🌐 Navegadores</h2>
                <table>
                    <tr>
                        <th>Navegador</th>
                        <th>Uso</th>
                    </tr>
        """
        
        for browser, count in stats['browsers'].most_common():
            html += f"""
                    <tr>
                        <td>{browser}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="footer">
                <p>📧 Reporte generado automáticamente por GERARD Email Reporter</p>
                <p>Consultor GERARD - Asistente Especializado</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_no_data_report(self, date: datetime) -> str:
        """Genera un reporte cuando no hay datos."""
        date_str = date.strftime("%d/%m/%Y")
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    text-align: center;
                    padding: 50px;
                    background-color: #f4f4f4;
                }}
                .message {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    max-width: 500px;
                    margin: 0 auto;
                }}
                h1 {{ color: #8A2BE2; }}
                p {{ color: #666; font-size: 1.1em; }}
            </style>
        </head>
        <body>
            <div class="message">
                <h1>📊 Reporte Diario</h1>
                <p><strong>{date_str}</strong></p>
                <p>No se registraron interacciones en esta fecha.</p>
            </div>
        </body>
        </html>
        """
    
    def send_email(self, html_content: str, date: datetime = None) -> bool:
        """
        Envía el reporte por email.
        
        Args:
            html_content: Contenido HTML del reporte
            date: Fecha del reporte
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        date_str = date.strftime("%d/%m/%Y")
        
        try:
            # Crear mensaje
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config['sender_email']
            msg['To'] = self.config['recipient_email']
            msg['Subject'] = f"{self.config['subject_prefix']} - {date_str}"
            
            # Adjuntar HTML
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # Conectar y enviar
            print(f"Conectando a {self.config['smtp_server']}...")
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            
            print(f"Autenticando como {self.config['sender_email']}...")
            server.login(self.config['sender_email'], self.config['sender_password'])
            
            print(f"Enviando email a {self.config['recipient_email']}...")
            server.send_message(msg)
            server.quit()
            
            print("✅ Email enviado exitosamente!")
            return True
            
        except Exception as e:
            print(f"❌ Error enviando email: {e}")
            return False


def main():
    """Función principal para ejecutar manualmente."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generador de reportes por email para GERARD')
    parser.add_argument('--date', type=str, help='Fecha del reporte (YYYYMMDD)', default=None)
    parser.add_argument('--preview', action='store_true', help='Solo generar y mostrar el HTML sin enviar')
    
    args = parser.parse_args()
    
    # Parsear fecha si se proporciona
    report_date = None
    if args.date:
        try:
            report_date = datetime.strptime(args.date, "%Y%m%d")
        except ValueError:
            print("❌ Formato de fecha inválido. Use YYYYMMDD")
            return
    
    reporter = EmailReporter()
    
    print("📊 Generando reporte...")
    html = reporter.generate_daily_report(report_date)
    
    if args.preview:
        # Guardar preview
        preview_file = "reporte_preview.html"
        with open(preview_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Preview guardado en: {preview_file}")
    else:
        # Enviar por email
        success = reporter.send_email(html, report_date)
        if success:
            print("✅ Reporte enviado exitosamente!")
        else:
            print("❌ Error enviando reporte")


if __name__ == "__main__":
    main()

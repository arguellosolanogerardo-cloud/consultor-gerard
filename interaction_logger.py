"""
Sistema de Logging Completo para Interacciones con GERARD

Este módulo captura y registra todas las interacciones de preguntas y respuestas,
incluyendo métricas de rendimiento detalladas, información del usuario, dispositivo,
ubicación geográfica y más.

Características:
- Timers de alta precisión (microsegundos)
- Captura de todas las fases de procesamiento
- Formato legible y estructurado
- Rotación automática de archivos
- Manejo robusto de errores
- Anonimización opcional de datos sensibles
"""

import time
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import traceback

from geo_utils import GeoLocator
from device_detector import DeviceDetector


class InteractionLogger:
    """
    Clase principal para el registro de interacciones.
    Captura información completa de cada consulta y respuesta.
    """
    
    def __init__(
        self,
        platform: str = "web",
        log_dir: str = "logs",
        anonymize: bool = False,
        max_file_size_mb: int = 10,
        enable_json: bool = True
    ):
        """
        Inicializa el logger.
        
        Args:
            platform: "web" o "terminal"
            log_dir: Directorio donde se guardarán los logs
            anonymize: Si True, anonimiza IPs y datos sensibles
            max_file_size_mb: Tamaño máximo del archivo antes de rotar
            enable_json: Si True, también guarda en formato JSON
        """
        self.platform = platform
        self.log_dir = Path(log_dir)
        self.anonymize = anonymize
        self.max_file_size_mb = max_file_size_mb
        self.enable_json = enable_json
        
        # Crear directorio de logs si no existe
        self.log_dir.mkdir(exist_ok=True)
        
        # Inicializar utilidades
        self.geo_locator = GeoLocator()
        self.device_detector = DeviceDetector()
        
        # Almacenamiento temporal de interacciones en curso
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Contador de registros del día
        self.daily_counter = self._get_daily_counter()
    
    def _get_daily_counter(self) -> int:
        """Obtiene el contador de registros del día actual."""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"interaction_log_{today}.txt"
        
        if not log_file.exists():
            return 0
        
        # Contar líneas que empiezan con "REGISTRO #"
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return content.count("REGISTRO #")
        except Exception:
            return 0
    
    def _get_log_filename(self, extension: str = "txt") -> Path:
        """Genera el nombre del archivo de log actual."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"interaction_log_{today}.{extension}"
    
    def _should_rotate(self, filepath: Path) -> bool:
        """Verifica si el archivo debe rotarse por tamaño."""
        if not filepath.exists():
            return False
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        return size_mb >= self.max_file_size_mb
    
    def _rotate_file(self, filepath: Path):
        """Rota un archivo agregando un sufijo numérico."""
        if not filepath.exists():
            return
        
        base = filepath.stem
        ext = filepath.suffix
        counter = 1
        
        while True:
            new_path = filepath.parent / f"{base}_part{counter}{ext}"
            if not new_path.exists():
                filepath.rename(new_path)
                break
            counter += 1
    
    def _anonymize_ip(self, ip: str) -> str:
        """Anonimiza una dirección IP usando hash."""
        if not ip or ip == "Desconocido":
            return "Anónimo"
        return hashlib.sha256(ip.encode()).hexdigest()[:16]
    
    def start_interaction(
        self,
        user: str,
        question: str,
        request_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Inicia el registro de una nueva interacción.
        
        Args:
            user: Nombre del usuario
            question: Pregunta realizada
            request_info: Información adicional (headers, etc.)
        
        Returns:
            session_id: ID único de la sesión
        """
        # Generar ID único de sesión
        session_id = f"{int(time.time() * 1000000)}"
        
        # Inicializar datos de la sesión
        session_data = {
            "session_id": session_id,
            "user": user,
            "question": question,
            "platform": self.platform,
            "timestamp_start": time.perf_counter(),
            "datetime_start": datetime.now(),
            "phases": {},
            "request_info": request_info or {}
        }
        
        # Capturar información del dispositivo y ubicación
        try:
            if self.platform == "web":
                user_agent = request_info.get("user_agent", "") if request_info else ""
                session_data["device_info"] = self.device_detector.detect_from_web(user_agent)
            else:
                session_data["device_info"] = self.device_detector.detect_from_terminal()
            
            # Obtener información geográfica
            session_data["geo_info"] = self.geo_locator.get_location()
            
            # Anonimizar IP si está configurado
            if self.anonymize and "ip" in session_data["geo_info"]:
                session_data["geo_info"]["ip"] = self._anonymize_ip(
                    session_data["geo_info"]["ip"]
                )
        
        except Exception as e:
            session_data["device_info"] = {"error": str(e)}
            session_data["geo_info"] = {"error": str(e)}
        
        # Marcar fase inicial
        session_data["phases"]["start"] = time.perf_counter()
        
        # Guardar sesión activa
        self.active_sessions[session_id] = session_data
        
        return session_id
    
    def mark_phase(self, session_id: str, phase_name: str):
        """
        Marca una fase específica del procesamiento.
        
        Args:
            session_id: ID de la sesión
            phase_name: Nombre de la fase (ej: "rag_start", "llm_start", etc.)
        """
        if session_id not in self.active_sessions:
            return
        
        self.active_sessions[session_id]["phases"][phase_name] = time.perf_counter()
    
    def log_response(
        self,
        session_id: str,
        answer: str,
        sources: Optional[List[Any]] = None,
        tokens: Optional[int] = None
    ):
        """
        Registra la respuesta generada.
        
        Args:
            session_id: ID de la sesión
            answer: Respuesta generada
            sources: Documentos fuente utilizados
            tokens: Número de tokens procesados (si está disponible)
        """
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session["answer"] = answer
        session["sources_count"] = len(sources) if sources else 0
        session["tokens"] = tokens
    
    def end_interaction(
        self,
        session_id: str,
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Finaliza y guarda el registro de la interacción.
        
        Args:
            session_id: ID de la sesión
            status: "success" o "error"
            error: Mensaje de error si aplica
        """
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session["phases"]["end"] = time.perf_counter()
        session["status"] = status
        session["error"] = error
        
        # Calcular métricas de tiempo
        metrics = self._calculate_metrics(session)
        session["metrics"] = metrics
        
        # Incrementar contador diario
        self.daily_counter += 1
        
        # Guardar en archivos
        try:
            self._save_to_txt(session, self.daily_counter)
            
            if self.enable_json:
                self._save_to_json(session)
        
        except Exception as e:
            # Guardar error en log de errores
            self._log_error(session_id, e)
        
        # Limpiar sesión activa
        del self.active_sessions[session_id]
    
    def _calculate_metrics(self, session: Dict[str, Any]) -> Dict[str, float]:
        """Calcula las métricas de tiempo de la sesión."""
        phases = session["phases"]
        start = phases.get("start", 0)
        end = phases.get("end", start)
        
        metrics = {
            "tiempo_total": end - start
        }
        
        # Calcular tiempos por fase
        phase_names = [
            "rag_start", "rag_end",
            "llm_start", "llm_end",
            "processing_start", "processing_end",
            "render_start", "render_end"
        ]
        
        prev_time = start
        for phase in phase_names:
            if phase in phases:
                metrics[f"tiempo_{phase}"] = phases[phase] - prev_time
                prev_time = phases[phase]
        
        # Calcular tiempos agregados por categoría
        if "rag_start" in phases and "rag_end" in phases:
            metrics["tiempo_rag"] = phases["rag_end"] - phases["rag_start"]
        
        if "llm_start" in phases and "llm_end" in phases:
            metrics["tiempo_llm"] = phases["llm_end"] - phases["llm_start"]
        
        if "processing_start" in phases and "processing_end" in phases:
            metrics["tiempo_procesamiento"] = phases["processing_end"] - phases["processing_start"]
        
        if "render_start" in phases and "render_end" in phases:
            metrics["tiempo_render"] = phases["render_end"] - phases["render_start"]
        
        return metrics
    
    def _save_to_txt(self, session: Dict[str, Any], counter: int):
        """Guarda el registro en formato de texto legible."""
        log_file = self._get_log_filename("txt")
        
        # Verificar si necesita rotación
        if self._should_rotate(log_file):
            self._rotate_file(log_file)
        
        # Formatear registro
        content = self._format_txt_log(session, counter)
        
        # Escribir en archivo
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
    
    def _format_txt_log(self, session: Dict[str, Any], counter: int) -> str:
        """Formatea el registro en texto legible."""
        dt = session["datetime_start"]
        device = session.get("device_info", {})
        geo = session.get("geo_info", {})
        metrics = session.get("metrics", {})
        
        # Construir línea de dispositivo
        device_line = f"{device.get('tipo', 'Desconocido')} - {device.get('os', 'Desconocido')}"
        
        # Construir línea de navegador/terminal
        if self.platform == "web":
            browser_line = f"Navegador: {device.get('navegador', 'Desconocido')}"
        else:
            browser_line = f"Terminal: {device.get('shell', 'Desconocido')}"
        
        # Construir línea de URL/ruta
        if self.platform == "web":
            url_line = f"URL: {session['request_info'].get('url', 'N/A')}"
        else:
            url_line = f"Directorio: {os.getcwd()}"
        
        # Estado con emoji
        status_emoji = "✅" if session["status"] == "success" else "❌"
        status_text = "Exitoso" if session["status"] == "success" else f"Error: {session.get('error', 'Desconocido')}"
        
        log_text = f"""=====================================
REGISTRO #{counter:03d}
Fecha/Hora: {dt.strftime("%Y-%m-%d %H:%M:%S")}
Usuario: {session['user']}
País: {geo.get('pais', 'Desconocido')}
Ciudad: {geo.get('ciudad', 'Desconocido')}
Coordenadas: {geo.get('coordenadas', 'N/A')}
IP: {geo.get('ip', 'Desconocido')}
Dispositivo: {device_line}
{browser_line}
Resolución: {device.get('resolucion', 'N/A')}
Plataforma: {self.platform.capitalize()}
{url_line}

PREGUNTA:
{session['question']}

RESPUESTA:
{session.get('answer', '[Sin respuesta registrada]')}

MÉTRICAS DE RENDIMIENTO:
─────────────────────────
⏱️ Tiempo total: {metrics.get('tiempo_total', 0):.3f}s
⏱️ Tiempo preparación RAG: {metrics.get('tiempo_rag', 0):.3f}s
⏱️ Tiempo consulta LLM: {metrics.get('tiempo_llm', 0):.3f}s
⏱️ Tiempo post-procesamiento: {metrics.get('tiempo_procesamiento', 0):.3f}s
⏱️ Tiempo renderizado: {metrics.get('tiempo_render', 0):.3f}s
⏱️ Latencia percibida: {metrics.get('tiempo_total', 0):.3f}s
📊 Tokens procesados: {session.get('tokens', 'N/A')}
📄 Documentos recuperados: {session.get('sources_count', 0)}
{status_emoji} Estado: {status_text}

=====================================
"""
        return log_text
    
    def _save_to_json(self, session: Dict[str, Any]):
        """Guarda el registro en formato JSON."""
        log_file = self._get_log_filename("json")
        
        # Preparar datos para JSON
        json_data = {
            "session_id": session["session_id"],
            "timestamp": session["datetime_start"].isoformat(),
            "user": session["user"],
            "platform": session["platform"],
            "question": session["question"],
            "answer": session.get("answer", ""),
            "device_info": session.get("device_info", {}),
            "geo_info": session.get("geo_info", {}),
            "metrics": session.get("metrics", {}),
            "sources_count": session.get("sources_count", 0),
            "tokens": session.get("tokens"),
            "status": session["status"],
            "error": session.get("error")
        }
        
        # Leer archivo existente o crear lista vacía
        data_list = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
            except Exception:
                data_list = []
        
        # Agregar nuevo registro
        data_list.append(json_data)
        
        # Guardar archivo actualizado
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    def _log_error(self, session_id: str, error: Exception):
        """Registra un error en el log de errores."""
        today = datetime.now().strftime("%Y-%m-%d")
        error_log = self.log_dir / f"error_log_{today}.txt"
        
        error_content = f"""
{'='*50}
Error en sesión: {session_id}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Error: {str(error)}
Traceback:
{traceback.format_exc()}
{'='*50}
"""
        
        with open(error_log, 'a', encoding='utf-8') as f:
            f.write(error_content)
    
    def generate_daily_summary(self, date: Optional[str] = None):
        """
        Genera un resumen estadístico del día.
        
        Args:
            date: Fecha en formato YYYY-MM-DD. Si None, usa hoy.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        json_file = self.log_dir / f"interaction_log_{date}.json"
        
        if not json_file.exists():
            print(f"No hay datos para la fecha {date}")
            return
        
        # Leer datos
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("No hay registros para analizar")
            return
        
        # Calcular estadísticas
        total_interactions = len(data)
        successful = sum(1 for d in data if d["status"] == "success")
        failed = total_interactions - successful
        
        times = [d["metrics"].get("tiempo_total", 0) for d in data if "metrics" in d]
        avg_time = sum(times) / len(times) if times else 0
        max_time = max(times) if times else 0
        min_time = min(times) if times else 0
        
        # Top 10 consultas más lentas
        sorted_data = sorted(data, key=lambda x: x["metrics"].get("tiempo_total", 0), reverse=True)
        slowest = sorted_data[:10]
        
        # Distribución por país
        countries = {}
        for d in data:
            country = d.get("geo_info", {}).get("pais", "Desconocido")
            countries[country] = countries.get(country, 0) + 1
        
        # Distribución por dispositivo
        devices = {}
        for d in data:
            device = d.get("device_info", {}).get("tipo", "Desconocido")
            devices[device] = devices.get(device, 0) + 1
        
        # Generar reporte
        summary_file = self.log_dir / f"performance_summary_{date}.txt"
        
        summary_content = f"""
═══════════════════════════════════════════════════════════
RESUMEN ESTADÍSTICO - {date}
═══════════════════════════════════════════════════════════

📊 ESTADÍSTICAS GENERALES
─────────────────────────
Total de interacciones: {total_interactions}
Exitosas: {successful} ({successful/total_interactions*100:.1f}%)
Fallidas: {failed} ({failed/total_interactions*100:.1f}%)

⏱️ MÉTRICAS DE RENDIMIENTO
─────────────────────────
Tiempo promedio: {avg_time:.3f}s
Tiempo mínimo: {min_time:.3f}s
Tiempo máximo: {max_time:.3f}s

🐌 TOP 10 CONSULTAS MÁS LENTAS
─────────────────────────────────
"""
        
        for i, item in enumerate(slowest, 1):
            time_val = item["metrics"].get("tiempo_total", 0)
            user = item.get("user", "N/A")
            question_preview = item.get("question", "")[:60] + "..."
            summary_content += f"{i}. {time_val:.3f}s - Usuario: {user}\n   Pregunta: {question_preview}\n\n"
        
        summary_content += """
🌍 DISTRIBUCIÓN POR PAÍS
─────────────────────────
"""
        for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_interactions * 100
            summary_content += f"{country}: {count} ({percentage:.1f}%)\n"
        
        summary_content += """
💻 DISTRIBUCIÓN POR DISPOSITIVO
────────────────────────────────
"""
        for device, count in sorted(devices.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_interactions * 100
            summary_content += f"{device}: {count} ({percentage:.1f}%)\n"
        
        summary_content += "\n═══════════════════════════════════════════════════════════\n"
        
        # Guardar resumen
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"Resumen generado: {summary_file}")
        print(summary_content)

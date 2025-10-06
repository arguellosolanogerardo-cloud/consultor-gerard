"""
Script para analizar estadísticas de los logs de interacciones con GERARD.

Este script lee los archivos de log JSON y genera estadísticas detalladas
sobre el uso del sistema, tiempos de respuesta, usuarios más activos, etc.

Uso:
    python analyze_logs.py [fecha]
    
    - Si no se especifica fecha, analiza el día actual
    - Formato de fecha: YYYY-MM-DD
    
Ejemplo:
    python analyze_logs.py 2025-10-06
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import Counter


class LogAnalyzer:
    """Analizador de logs de interacciones."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
    
    def analyze_date(self, date: str = None):
        """
        Analiza los logs de una fecha específica.
        
        Args:
            date: Fecha en formato YYYY-MM-DD. Si None, usa hoy.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        json_file = self.log_dir / f"interaction_log_{date}.json"
        
        if not json_file.exists():
            print(f"❌ No se encontraron logs para la fecha {date}")
            print(f"   Buscando en: {json_file}")
            return
        
        # Leer datos
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print(f"⚠️  El archivo de logs está vacío para {date}")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 ANÁLISIS DE LOGS - {date}")
        print(f"{'='*70}\n")
        
        self._print_general_stats(data)
        self._print_performance_stats(data)
        self._print_user_stats(data)
        self._print_geographic_stats(data)
        self._print_device_stats(data)
        self._print_slowest_queries(data)
        self._print_error_stats(data)
        
        print(f"\n{'='*70}\n")
    
    def _print_general_stats(self, data: List[Dict]):
        """Imprime estadísticas generales."""
        total = len(data)
        successful = sum(1 for d in data if d.get("status") == "success")
        failed = total - successful
        
        print(f"📈 ESTADÍSTICAS GENERALES")
        print(f"{'-'*70}")
        print(f"Total de interacciones: {total}")
        print(f"  ✅ Exitosas: {successful} ({successful/total*100:.1f}%)")
        print(f"  ❌ Fallidas: {failed} ({failed/total*100:.1f}%)")
        print()
    
    def _print_performance_stats(self, data: List[Dict]):
        """Imprime estadísticas de rendimiento."""
        times = [d["metrics"].get("tiempo_total", 0) for d in data if "metrics" in d]
        
        if not times:
            return
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # Calcular percentiles
        sorted_times = sorted(times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        print(f"⚡ MÉTRICAS DE RENDIMIENTO")
        print(f"{'-'*70}")
        print(f"Tiempo promedio de respuesta: {avg_time:.3f}s")
        print(f"Tiempo mínimo: {min_time:.3f}s")
        print(f"Tiempo máximo: {max_time:.3f}s")
        print(f"Mediana (P50): {p50:.3f}s")
        print(f"Percentil 95 (P95): {p95:.3f}s")
        print(f"Percentil 99 (P99): {p99:.3f}s")
        
        # Desglose de tiempos
        llm_times = [d["metrics"].get("tiempo_llm", 0) for d in data if "metrics" in d and "tiempo_llm" in d["metrics"]]
        if llm_times:
            avg_llm = sum(llm_times) / len(llm_times)
            print(f"\nTiempo promedio de LLM: {avg_llm:.3f}s ({avg_llm/avg_time*100:.1f}% del total)")
        
        print()
    
    def _print_user_stats(self, data: List[Dict]):
        """Imprime estadísticas por usuario."""
        users = Counter([d.get("user", "Desconocido") for d in data])
        
        print(f"👥 USUARIOS MÁS ACTIVOS")
        print(f"{'-'*70}")
        for i, (user, count) in enumerate(users.most_common(10), 1):
            percentage = count / len(data) * 100
            print(f"{i:2d}. {user}: {count} consultas ({percentage:.1f}%)")
        print()
    
    def _print_geographic_stats(self, data: List[Dict]):
        """Imprime estadísticas geográficas."""
        countries = Counter([
            d.get("geo_info", {}).get("pais", "Desconocido") 
            for d in data
        ])
        
        cities = Counter([
            f"{d.get('geo_info', {}).get('ciudad', 'Desconocido')}, {d.get('geo_info', {}).get('pais', 'Desconocido')}"
            for d in data
        ])
        
        print(f"🌍 DISTRIBUCIÓN GEOGRÁFICA")
        print(f"{'-'*70}")
        print("Por país:")
        for i, (country, count) in enumerate(countries.most_common(5), 1):
            percentage = count / len(data) * 100
            print(f"  {i}. {country}: {count} ({percentage:.1f}%)")
        
        print("\nCiudades principales:")
        for i, (city, count) in enumerate(cities.most_common(5), 1):
            percentage = count / len(data) * 100
            print(f"  {i}. {city}: {count} ({percentage:.1f}%)")
        print()
    
    def _print_device_stats(self, data: List[Dict]):
        """Imprime estadísticas de dispositivos."""
        platforms = Counter([d.get("platform", "Desconocido") for d in data])
        device_types = Counter([
            d.get("device_info", {}).get("tipo", "Desconocido")
            for d in data
        ])
        browsers = Counter([
            d.get("device_info", {}).get("navegador", "N/A")
            for d in data
            if d.get("platform") == "web"
        ])
        
        print(f"💻 DISPOSITIVOS Y PLATAFORMAS")
        print(f"{'-'*70}")
        print("Por plataforma:")
        for platform, count in platforms.most_common():
            percentage = count / len(data) * 100
            print(f"  {platform}: {count} ({percentage:.1f}%)")
        
        print("\nTipo de dispositivo:")
        for device_type, count in device_types.most_common():
            percentage = count / len(data) * 100
            print(f"  {device_type}: {count} ({percentage:.1f}%)")
        
        if browsers:
            print("\nNavegadores (solo web):")
            for i, (browser, count) in enumerate(browsers.most_common(5), 1):
                print(f"  {i}. {browser}: {count}")
        print()
    
    def _print_slowest_queries(self, data: List[Dict], n: int = 10):
        """Imprime las consultas más lentas."""
        sorted_data = sorted(
            data,
            key=lambda x: x["metrics"].get("tiempo_total", 0),
            reverse=True
        )
        
        print(f"🐌 TOP {n} CONSULTAS MÁS LENTAS")
        print(f"{'-'*70}")
        for i, item in enumerate(sorted_data[:n], 1):
            time_val = item["metrics"].get("tiempo_total", 0)
            user = item.get("user", "N/A")
            question = item.get("question", "")
            question_preview = question[:60] + "..." if len(question) > 60 else question
            timestamp = datetime.fromisoformat(item.get("timestamp", "")).strftime("%H:%M:%S")
            
            print(f"{i:2d}. {time_val:6.3f}s - {timestamp} - {user}")
            print(f"    Q: {question_preview}")
        print()
    
    def _print_error_stats(self, data: List[Dict]):
        """Imprime estadísticas de errores."""
        errors = [d for d in data if d.get("status") == "error"]
        
        if not errors:
            print(f"✅ NO SE REGISTRARON ERRORES")
            print(f"{'-'*70}\n")
            return
        
        error_types = Counter([
            d.get("error", "Error desconocido")[:100]
            for d in errors
        ])
        
        print(f"⚠️  ERRORES REGISTRADOS ({len(errors)} total)")
        print(f"{'-'*70}")
        for i, (error, count) in enumerate(error_types.most_common(), 1):
            print(f"{i}. {error}")
            print(f"   Ocurrencias: {count}")
        print()
    
    def list_available_dates(self):
        """Lista todas las fechas disponibles en los logs."""
        json_files = list(self.log_dir.glob("interaction_log_*.json"))
        
        if not json_files:
            print("❌ No se encontraron archivos de log")
            return
        
        print(f"\n📅 FECHAS DISPONIBLES:")
        print(f"{'-'*70}")
        
        dates_info = []
        for file in sorted(json_files):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    date = file.stem.replace("interaction_log_", "")
                    count = len(data)
                    dates_info.append((date, count))
            except Exception:
                pass
        
        for date, count in dates_info:
            print(f"  {date}: {count} interacciones")
        print()


def main():
    """Función principal."""
    analyzer = LogAnalyzer()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            analyzer.list_available_dates()
        else:
            date = sys.argv[1]
            analyzer.analyze_date(date)
    else:
        # Analizar hoy por defecto
        analyzer.analyze_date()


if __name__ == "__main__":
    main()

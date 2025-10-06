"""
Módulo de Geolocalización para el Sistema de Logging

Este módulo proporciona funcionalidades para obtener información geográfica
del usuario basada en su dirección IP.

Características:
- Múltiples APIs de respaldo (ipapi.co, ip-api.com, ipinfo.io)
- Cache de resultados para evitar llamadas repetidas
- Manejo robusto de errores
- Timeout configurable
- Detección de IP pública
"""

import requests
import socket
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path


class GeoLocator:
    """
    Clase para obtener información geográfica basada en IP.
    """
    
    def __init__(
        self,
        cache_duration_minutes: int = 60,
        timeout_seconds: int = 5
    ):
        """
        Inicializa el geolocalizador.
        
        Args:
            cache_duration_minutes: Duración del cache en minutos
            timeout_seconds: Timeout para las peticiones HTTP
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.timeout = timeout_seconds
        self.cache_file = Path("logs/.geo_cache.json")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Carga el cache desde el archivo."""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                # Limpiar entradas expiradas
                now = datetime.now()
                valid_cache = {}
                for key, value in cache.items():
                    if 'timestamp' in value:
                        cached_time = datetime.fromisoformat(value['timestamp'])
                        if now - cached_time < self.cache_duration:
                            valid_cache[key] = value
                return valid_cache
        except Exception:
            return {}
    
    def _save_cache(self):
        """Guarda el cache en el archivo."""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass
    
    def _get_public_ip(self) -> Optional[str]:
        """
        Obtiene la dirección IP pública del usuario.
        
        Returns:
            IP pública o None si no se puede obtener
        """
        # Lista de servicios para obtener la IP
        ip_services = [
            "https://api.ipify.org?format=json",
            "https://ipapi.co/json",
            "http://ip-api.com/json/"
        ]
        
        for service in ip_services:
            try:
                response = requests.get(service, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    # Diferentes servicios devuelven la IP con diferentes claves
                    if 'ip' in data:
                        return data['ip']
                    elif 'query' in data:
                        return data['query']
            except Exception:
                continue
        
        # Si todo falla, intentar obtener IP local (no será pública)
        try:
            # Truco para obtener la IP local sin conectar realmente
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return None
    
    def _get_location_from_ipapi_co(self, ip: str) -> Optional[Dict]:
        """
        Obtiene ubicación usando ipapi.co (gratuito, 1000 req/día).
        
        Args:
            ip: Dirección IP
        
        Returns:
            Diccionario con información geográfica o None
        """
        try:
            url = f"https://ipapi.co/{ip}/json/"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                # Verificar si hay error
                if 'error' in data:
                    return None
                
                return {
                    "ip": ip,
                    "pais": data.get("country_name", "Desconocido"),
                    "ciudad": data.get("city", "Desconocido"),
                    "region": data.get("region", "Desconocido"),
                    "coordenadas": f"{data.get('latitude', 'N/A')}, {data.get('longitude', 'N/A')}",
                    "codigo_pais": data.get("country_code", "N/A"),
                    "timezone": data.get("timezone", "N/A"),
                    "org": data.get("org", "N/A"),
                    "fuente": "ipapi.co"
                }
        except Exception:
            return None
    
    def _get_location_from_ipapi_com(self, ip: str) -> Optional[Dict]:
        """
        Obtiene ubicación usando ip-api.com (gratuito, 45 req/min).
        
        Args:
            ip: Dirección IP
        
        Returns:
            Diccionario con información geográfica o None
        """
        try:
            url = f"http://ip-api.com/json/{ip}"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                # Verificar si hay error
                if data.get("status") == "fail":
                    return None
                
                return {
                    "ip": ip,
                    "pais": data.get("country", "Desconocido"),
                    "ciudad": data.get("city", "Desconocido"),
                    "region": data.get("regionName", "Desconocido"),
                    "coordenadas": f"{data.get('lat', 'N/A')}, {data.get('lon', 'N/A')}",
                    "codigo_pais": data.get("countryCode", "N/A"),
                    "timezone": data.get("timezone", "N/A"),
                    "org": data.get("isp", "N/A"),
                    "fuente": "ip-api.com"
                }
        except Exception:
            return None
    
    def _get_location_from_ipinfo_io(self, ip: str) -> Optional[Dict]:
        """
        Obtiene ubicación usando ipinfo.io (gratuito, 50k req/mes).
        
        Args:
            ip: Dirección IP
        
        Returns:
            Diccionario con información geográfica o None
        """
        try:
            url = f"https://ipinfo.io/{ip}/json"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                # Separar ciudad y región
                location_parts = data.get("loc", ",").split(",")
                city = data.get("city", "Desconocido")
                region = data.get("region", "Desconocido")
                
                return {
                    "ip": ip,
                    "pais": data.get("country", "Desconocido"),
                    "ciudad": city,
                    "region": region,
                    "coordenadas": data.get("loc", "N/A"),
                    "codigo_pais": data.get("country", "N/A"),
                    "timezone": data.get("timezone", "N/A"),
                    "org": data.get("org", "N/A"),
                    "fuente": "ipinfo.io"
                }
        except Exception:
            return None
    
    def get_location(self, ip: Optional[str] = None) -> Dict:
        """
        Obtiene la información geográfica del usuario.
        
        Args:
            ip: IP específica a consultar. Si None, obtiene la IP pública actual.
        
        Returns:
            Diccionario con información geográfica
        """
        # Si no se proporciona IP, obtener la pública
        if ip is None:
            ip = self._get_public_ip()
        
        if not ip:
            return {
                "ip": "Desconocido",
                "pais": "Desconocido",
                "ciudad": "Desconocido",
                "region": "Desconocido",
                "coordenadas": "N/A",
                "codigo_pais": "N/A",
                "timezone": "N/A",
                "org": "N/A",
                "error": "No se pudo obtener la IP"
            }
        
        # Verificar cache
        if ip in self.cache:
            cached_data = self.cache[ip]
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time < self.cache_duration:
                # Retornar datos del cache (sin el timestamp)
                result = {k: v for k, v in cached_data.items() if k != 'timestamp'}
                return result
        
        # Intentar obtener ubicación con diferentes servicios
        location_data = None
        
        # Prioridad de servicios
        services = [
            self._get_location_from_ipapi_co,
            self._get_location_from_ipapi_com,
            self._get_location_from_ipinfo_io
        ]
        
        for service in services:
            location_data = service(ip)
            if location_data:
                break
        
        # Si no se pudo obtener, retornar datos básicos
        if not location_data:
            location_data = {
                "ip": ip,
                "pais": "Desconocido",
                "ciudad": "Desconocido",
                "region": "Desconocido",
                "coordenadas": "N/A",
                "codigo_pais": "N/A",
                "timezone": "N/A",
                "org": "N/A",
                "error": "No se pudo obtener información geográfica"
            }
        
        # Guardar en cache con timestamp
        cache_data = location_data.copy()
        cache_data['timestamp'] = datetime.now().isoformat()
        self.cache[ip] = cache_data
        self._save_cache()
        
        return location_data
    
    def get_location_by_hostname(self, hostname: str) -> Dict:
        """
        Obtiene la ubicación basada en un hostname.
        
        Args:
            hostname: Nombre del host (ej: "google.com")
        
        Returns:
            Diccionario con información geográfica
        """
        try:
            ip = socket.gethostbyname(hostname)
            return self.get_location(ip)
        except Exception as e:
            return {
                "ip": "Desconocido",
                "pais": "Desconocido",
                "ciudad": "Desconocido",
                "region": "Desconocido",
                "coordenadas": "N/A",
                "codigo_pais": "N/A",
                "timezone": "N/A",
                "org": "N/A",
                "error": f"No se pudo resolver el hostname: {str(e)}"
            }
    
    def is_local_ip(self, ip: str) -> bool:
        """
        Verifica si una IP es local/privada.
        
        Args:
            ip: Dirección IP a verificar
        
        Returns:
            True si es IP local, False si es pública
        """
        if not ip:
            return False
        
        # Rangos de IPs privadas
        private_ranges = [
            "127.",  # Loopback
            "10.",   # Clase A privada
            "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.",
            "172.24.", "172.25.", "172.26.", "172.27.",
            "172.28.", "172.29.", "172.30.", "172.31.",  # Clase B privada
            "192.168.",  # Clase C privada
            "169.254."   # Link-local
        ]
        
        return any(ip.startswith(prefix) for prefix in private_ranges)
    
    def clear_cache(self):
        """Limpia el cache de geolocalización."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


# Función de conveniencia para uso rápido
def get_current_location() -> Dict:
    """
    Obtiene la ubicación geográfica actual de forma rápida.
    
    Returns:
        Diccionario con información geográfica
    """
    locator = GeoLocator()
    return locator.get_location()


# Prueba del módulo
if __name__ == "__main__":
    print("Probando GeoLocator...")
    locator = GeoLocator()
    
    print("\n1. Obteniendo ubicación actual:")
    location = locator.get_location()
    print(json.dumps(location, indent=2, ensure_ascii=False))
    
    print("\n2. Verificando cache (segunda llamada):")
    location2 = locator.get_location()
    print(json.dumps(location2, indent=2, ensure_ascii=False))
    
    print("\n3. Probando con IP específica (8.8.8.8 - Google DNS):")
    google_location = locator.get_location("8.8.8.8")
    print(json.dumps(google_location, indent=2, ensure_ascii=False))
    
    print("\n4. Probando detección de IP local:")
    print(f"¿127.0.0.1 es local? {locator.is_local_ip('127.0.0.1')}")
    print(f"¿8.8.8.8 es local? {locator.is_local_ip('8.8.8.8')}")

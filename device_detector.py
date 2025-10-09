"""
Módulo de Detección de Dispositivos para el Sistema de Logging

Este módulo detecta información del dispositivo y sistema del usuario,
tanto para entornos web como terminal.

Características:
- Análisis de User-Agent para entornos web
- Detección de sistema operativo
- Detección de navegador y versión
- Información de hardware cuando está disponible
- Detección de terminal y shell en entornos de consola
"""

import platform
import os
import shutil
from typing import Dict, Optional
import re


class DeviceDetector:
    """
    Clase para detectar información del dispositivo y sistema.
    """
    
    def __init__(self):
        """Inicializa el detector de dispositivos."""
        self.platform_info = self._get_platform_info()
    
    def _get_platform_info(self) -> Dict:
        """Obtiene información básica de la plataforma."""
        try:
            return {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            }
        except Exception:
            return {}
    
    def detect_from_web(self, user_agent: str) -> Dict:
        """
        Detecta información del dispositivo desde un User-Agent web.
        
        Args:
            user_agent: String del User-Agent
        
        Returns:
            Diccionario con información del dispositivo
        """
        if not user_agent:
            return self._get_default_web_info()
        
        # Inicializar resultado
        result = {
            "tipo": "Desconocido",
            "os": "Desconocido",
            "os_version": "N/A",
            "navegador": "Desconocido",
            "navegador_version": "N/A",
            "resolucion": "N/A",
            "user_agent": user_agent
        }
        
        # Detectar tipo de dispositivo
        result["tipo"] = self._detect_device_type(user_agent)
        
        # Detectar sistema operativo
        os_info = self._detect_os_from_ua(user_agent)
        result["os"] = os_info["name"]
        result["os_version"] = os_info["version"]
        
        # Detectar navegador
        browser_info = self._detect_browser(user_agent)
        result["navegador"] = browser_info["name"]
        result["navegador_version"] = browser_info["version"]
        
        return result
    
    def detect_from_terminal(self) -> Dict:
        """
        Detecta información del dispositivo desde un entorno terminal.
        
        Returns:
            Diccionario con información del dispositivo
        """
        result = {
            "tipo": "PC",
            "os": platform.system(),
            "os_version": platform.release(),
            "shell": self._detect_shell(),
            "terminal": self._detect_terminal(),
            "resolucion": self._get_terminal_size(),
            "arquitectura": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version()
        }
        
        # Información adicional según el SO
        if result["os"] == "Windows":
            result["os_details"] = self._get_windows_info()
        elif result["os"] == "Linux":
            result["os_details"] = self._get_linux_info()
        elif result["os"] == "Darwin":
            result["os"] = "macOS"
            result["os_details"] = self._get_macos_info()
        
        return result
    
    def _detect_device_type(self, user_agent: str) -> str:
        """Detecta el tipo de dispositivo desde el User-Agent."""
        ua_lower = user_agent.lower()
        
        # Móviles
        mobile_keywords = [
            'android', 'iphone', 'ipad', 'ipod', 'blackberry',
            'windows phone', 'mobile', 'webos', 'opera mini'
        ]
        if any(keyword in ua_lower for keyword in mobile_keywords):
            if 'ipad' in ua_lower or 'tablet' in ua_lower:
                return "Tablet"
            return "Móvil"
        
        # Tablets específicas
        tablet_keywords = ['tablet', 'kindle', 'silk', 'playbook']
        if any(keyword in ua_lower for keyword in tablet_keywords):
            return "Tablet"
        
        return "PC"
    
    def _detect_os_from_ua(self, user_agent: str) -> Dict:
        """Detecta el sistema operativo desde el User-Agent."""
        ua_lower = user_agent.lower()
        
        # Windows
        if 'windows nt 10.0' in ua_lower:
            return {"name": "Windows", "version": "10/11"}
        elif 'windows nt 6.3' in ua_lower:
            return {"name": "Windows", "version": "8.1"}
        elif 'windows nt 6.2' in ua_lower:
            return {"name": "Windows", "version": "8"}
        elif 'windows nt 6.1' in ua_lower:
            return {"name": "Windows", "version": "7"}
        elif 'windows' in ua_lower:
            return {"name": "Windows", "version": "Desconocido"}
        
        # macOS / iOS
        if 'mac os x' in ua_lower or 'macos' in ua_lower:
            version_match = re.search(r'mac os x ([\d_]+)', ua_lower)
            version = version_match.group(1).replace('_', '.') if version_match else "Desconocido"
            return {"name": "macOS", "version": version}
        elif 'iphone os' in ua_lower or 'ios' in ua_lower:
            version_match = re.search(r'os ([\d_]+)', ua_lower)
            version = version_match.group(1).replace('_', '.') if version_match else "Desconocido"
            return {"name": "iOS", "version": version}
        
        # Android
        if 'android' in ua_lower:
            version_match = re.search(r'android ([\d.]+)', ua_lower)
            version = version_match.group(1) if version_match else "Desconocido"
            return {"name": "Android", "version": version}
        
        # Linux
        if 'linux' in ua_lower:
            if 'ubuntu' in ua_lower:
                return {"name": "Ubuntu", "version": "Desconocido"}
            elif 'fedora' in ua_lower:
                return {"name": "Fedora", "version": "Desconocido"}
            return {"name": "Linux", "version": "Desconocido"}
        
        return {"name": "Desconocido", "version": "N/A"}
    
    def _detect_browser(self, user_agent: str) -> Dict:
        """Detecta el navegador desde el User-Agent."""
        ua = user_agent
        
        # Edge (debe ir antes de Chrome)
        if 'Edg/' in ua:
            version_match = re.search(r'Edg/([\d.]+)', ua)
            version = version_match.group(1) if version_match else "Desconocido"
            return {"name": "Edge", "version": version}
        
        # Chrome (debe ir antes de Safari)
        if 'Chrome/' in ua and 'Safari/' in ua:
            version_match = re.search(r'Chrome/([\d.]+)', ua)
            version = version_match.group(1) if version_match else "Desconocido"
            return {"name": "Chrome", "version": version}
        
        # Firefox
        if 'Firefox/' in ua:
            version_match = re.search(r'Firefox/([\d.]+)', ua)
            version = version_match.group(1) if version_match else "Desconocido"
            return {"name": "Firefox", "version": version}
        
        # Safari (debe ir después de Chrome)
        if 'Safari/' in ua:
            version_match = re.search(r'Version/([\d.]+)', ua)
            version = version_match.group(1) if version_match else "Desconocido"
            return {"name": "Safari", "version": version}
        
        # Opera
        if 'OPR/' in ua or 'Opera/' in ua:
            version_match = re.search(r'(?:OPR|Opera)/([\d.]+)', ua)
            version = version_match.group(1) if version_match else "Desconocido"
            return {"name": "Opera", "version": version}
        
        # Internet Explorer
        if 'MSIE' in ua or 'Trident/' in ua:
            version_match = re.search(r'(?:MSIE |rv:)([\d.]+)', ua)
            version = version_match.group(1) if version_match else "Desconocido"
            return {"name": "Internet Explorer", "version": version}
        
        return {"name": "Desconocido", "version": "N/A"}
    
    def _detect_shell(self) -> str:
        """Detecta el shell en uso."""
        shell = os.environ.get('SHELL', '')
        
        if shell:
            shell_name = os.path.basename(shell)
            return shell_name
        
        # En Windows, verificar variables específicas
        if platform.system() == "Windows":
            # PowerShell
            if os.environ.get('PSModulePath'):
                ps_version = os.environ.get('POWERSHELL_VERSION', '')
                if ps_version:
                    return f"PowerShell {ps_version}"
                return "PowerShell"
            # CMD
            return "CMD"
        
        return "Desconocido"
    
    def _detect_terminal(self) -> str:
        """Detecta el emulador de terminal en uso."""
        # Variables de entorno que indican el terminal
        term_env = os.environ.get('TERM', '')
        term_program = os.environ.get('TERM_PROGRAM', '')
        
        if term_program:
            return term_program
        
        if 'xterm' in term_env:
            return "XTerm"
        elif 'screen' in term_env:
            return "GNU Screen"
        elif term_env:
            return term_env
        
        # En Windows
        if platform.system() == "Windows":
            if os.environ.get('WT_SESSION'):
                return "Windows Terminal"
            return "Console"
        
        return "Desconocido"
    
    def _get_terminal_size(self) -> str:
        """Obtiene el tamaño del terminal."""
        try:
            size = shutil.get_terminal_size()
            return f"{size.columns}x{size.lines}"
        except Exception:
            return "N/A"
    
    def _get_windows_info(self) -> str:
        """Obtiene información adicional de Windows."""
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
            )
            product_name, _ = winreg.QueryValueEx(key, "ProductName")
            build, _ = winreg.QueryValueEx(key, "CurrentBuild")
            winreg.CloseKey(key)
            return f"{product_name} (Build {build})"
        except Exception:
            return platform.platform()
    
    def _get_linux_info(self) -> str:
        """Obtiene información adicional de Linux."""
        try:
            # Intentar leer /etc/os-release
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    lines = f.readlines()
                    info = {}
                    for line in lines:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            info[key] = value.strip('"')
                    
                    name = info.get('PRETTY_NAME', info.get('NAME', 'Linux'))
                    return name
        except Exception:
            pass
        
        return platform.platform()
    
    def _get_macos_info(self) -> str:
        """Obtiene información adicional de macOS."""
        try:
            import subprocess
            result = subprocess.run(
                ['sw_vers', '-productVersion'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return f"macOS {version}"
        except Exception:
            pass
        
        return platform.platform()
    
    def _get_default_web_info(self) -> Dict:
        """Retorna información por defecto cuando no hay User-Agent."""
        return {
            "tipo": "Desconocido",
            "os": "Desconocido",
            "os_version": "N/A",
            "navegador": "Desconocido",
            "navegador_version": "N/A",
            "resolucion": "N/A",
            "user_agent": "No disponible"
        }
    
    def get_screen_resolution(self) -> Optional[str]:
        """
        Intenta obtener la resolución de pantalla.
        Solo funciona en algunos entornos.
        
        Returns:
            String con la resolución o None
        """
        try:
            # En Windows
            if platform.system() == "Windows":
                import ctypes
                user32 = ctypes.windll.user32
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
                return f"{width}x{height}"
            
            # En Linux con X11
            elif platform.system() == "Linux":
                import subprocess
                result = subprocess.run(
                    ['xrandr'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    match = re.search(r'(\d+)x(\d+)\+', result.stdout)
                    if match:
                        return f"{match.group(1)}x{match.group(2)}"
            
            # En macOS
            elif platform.system() == "Darwin":
                import subprocess
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    match = re.search(r'Resolution: (\d+ x \d+)', result.stdout)
                    if match:
                        return match.group(1).replace(' ', '')
        
        except Exception:
            pass
        
        return None


# Función de conveniencia para uso rápido
def get_device_info() -> Dict:
    """
    Obtiene información del dispositivo actual de forma rápida.
    
    Returns:
        Diccionario con información del dispositivo
    """
    detector = DeviceDetector()
    return detector.detect_from_terminal()


# Prueba del módulo
if __name__ == "__main__":
    import json
    
    print("Probando DeviceDetector...")
    detector = DeviceDetector()
    
    print("\n1. Información del terminal actual:")
    terminal_info = detector.detect_from_terminal()
    print(json.dumps(terminal_info, indent=2, ensure_ascii=False))
    
    print("\n2. Probando detección desde User-Agent (Chrome en Windows):")
    ua_chrome = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    chrome_info = detector.detect_from_web(ua_chrome)
    print(json.dumps(chrome_info, indent=2, ensure_ascii=False))
    
    print("\n3. Probando detección desde User-Agent (Safari en iPhone):")
    ua_iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
    iphone_info = detector.detect_from_web(ua_iphone)
    print(json.dumps(iphone_info, indent=2, ensure_ascii=False))
    
    print("\n4. Probando detección desde User-Agent (Firefox en Linux):")
    ua_firefox = "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"
    firefox_info = detector.detect_from_web(ua_firefox)
    print(json.dumps(firefox_info, indent=2, ensure_ascii=False))
    
    print("\n5. Intentando obtener resolución de pantalla:")
    resolution = detector.get_screen_resolution()
    print(f"Resolución: {resolution if resolution else 'No disponible'}")

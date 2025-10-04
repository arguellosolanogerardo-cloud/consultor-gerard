#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = [
    ("GOOGLE_API_KEY assignment", re.compile(r"GOOGLE_API_KEY\s*=\s*['\"]?.{10,}['\"]?", re.IGNORECASE)),
    ("Google API key (AIza...)", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |)PRIVATE KEY-----")),
    ("AWS secret", re.compile(r"AWS_SECRET_ACCESS_KEY", re.IGNORECASE)),
    ("Generic secret var", re.compile(r"\b(secret|secret_key|client_secret|api_key|password)\b", re.IGNORECASE)),
    ("GitHub token", re.compile(r"ghp_[0-9A-Za-z_]{36,}|GITHUB_TOKEN", re.IGNORECASE)),
]

def get_tracked_files():
    p = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if p.returncode != 0:
        print("Error: no se pudo listar archivos git. Asegúrate de que git esté disponible.")
        sys.exit(1)
    return [Path(f) for f in p.stdout.splitlines() if f]

def is_binary(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            chunk = f.read(8000)
            return b'\0' in chunk
    except Exception:
        return True

def scan_file(path: Path):
    findings = []
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return findings

    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        for name, pattern in PATTERNS:
            if pattern.search(line):
                findings.append((i, name, line.strip()))
    return findings

def main():
    files = get_tracked_files()
    total_findings = 0
    report = []
    # Optionally ignore some files (the action and workflow themselves should be scanned too)
    for file in files:
        # skip very large files
        try:
            if file.stat().st_size > 5_000_000:
                continue
        except Exception:
            pass

        if is_binary(file):
            continue

        findings = scan_file(file)
        if findings:
            total_findings += len(findings)
            report.append((str(file), findings))

    if total_findings == 0:
        print("Secret scanner: no se detectaron posibles secretos en archivos rastreados.")
        return 0

    print("Secret scanner: SE DETECTARON POSIBLES SECRETOS. Revisa y corrige antes de empujar:\n")
    for file, findings in report:
        print(f"Archivo: {file}")
        for ln, name, snippet in findings:
            print(f"  Line {ln}: [{name}] {snippet}")
        print()

    # Targeted guidance
    print("Si alguno de estos hallazgos es una clave real, revoca/regenera la clave y elimínala del repositorio.\n")
    sys.exit(2)

if __name__ == '__main__':
    sys.exit(main())

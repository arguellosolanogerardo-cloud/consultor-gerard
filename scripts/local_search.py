import os
import re
import sys

SEARCH = sys.argv[1] if len(sys.argv) > 1 else 'eternidad'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRT_DIR = os.path.join(ROOT, 'documentos_srt')

pattern_timestamp = re.compile(r"(\d{2}:\d{2}:\d{2}),?\d{0,3}\s*-->\s*(\d{2}:\d{2}:\d{2}),?\d{0,3}")

matches = []

for fname in os.listdir(SRT_DIR):
    if not fname.lower().endswith('.srt'):
        continue
    path = os.path.join(SRT_DIR, fname)
    try:
        text = open(path, 'r', encoding='utf-8', errors='ignore').read()
    except Exception:
        continue

    # Split into blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        if SEARCH.lower() in block.lower():
            # try to find timestamp line
            ts_match = pattern_timestamp.search(block)
            ts = ts_match.group(0) if ts_match else 'Timestamp desconocido'
            # extract cleaned snippet: remove index lines and timestamp lines
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            snippet_lines = [l for l in lines if not re.match(r"^\d+$", l) and not pattern_timestamp.search(l)]
            snippet = ' '.join(snippet_lines)
            matches.append({'file': fname, 'timestamp': ts, 'snippet': snippet})

# Sort matches by file then timestamp (not exact chronological across files)

print(f"Encontradas {len(matches)} coincidencias para '{SEARCH}' (mostrando hasta 10):\n")
for m in matches[:10]:
    print(f"Archivo: {m['file']}")
    print(f"Timestamp: {m['timestamp']}")
    # highlight the search term in snippet (simple)
    highlighted = re.sub(re.escape(SEARCH), lambda mo: mo.group(0).upper(), m['snippet'], flags=re.IGNORECASE)
    print(f"Fragmento: {highlighted}")
    print('-' * 60)

if not matches:
    print('No se encontraron coincidencias locales.')
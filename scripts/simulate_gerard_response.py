import os
import re
import json
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRT_DIR = os.path.join(ROOT, 'documentos_srt')

SEARCH = 'eternidad'
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
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        if SEARCH.lower() in block.lower():
            ts_match = pattern_timestamp.search(block)
            ts = ts_match.group(0) if ts_match else '00:00:00 --> 00:00:00'
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            snippet_lines = [l for l in lines if not re.match(r"^\d+$", l) and not pattern_timestamp.search(l)]
            snippet = ' '.join(snippet_lines)
            matches.append({'file': fname, 'timestamp': ts, 'snippet': snippet})

# Take top 3 matches
top = matches[:3]

# Craft a JSON array following GERARD format
response = []

# Intro
response.append({
    "type": "normal",
    "content": "La eternidad, según los textos recuperados en el corpus, se describe como una cualidad de la vida que trasciende el tiempo y que está íntimamente ligada a la presencia del Padre/energía creadora." 
})

# Emphasis key idea
response.append({"type": "emphasis", "content": "la continuidad de la vida; la ETERNIDAD como estado de ser"})

# More normal
response.append({"type": "normal", "content": ", una realidad que los maestros mencionan como 'vida', 'luz' y 'permanencia'."})

# Add citations from top matches
for m in top:
    src = m['file']
    ts = m['timestamp']
    snippet = m['snippet']
    response.append({
        "type": "normal",
        "content": f"Cita: \"{snippet}\" (Fuente: {src}, Timestamp: {ts})"
    })

# Final suggestion
response.append({"type": "normal", "content": "En resumen: la eternidad se presenta como la continuación esencial de la vida y la presencia del Amor/Padre, no como un simple 'tiempo infinito' sino como cualidad y condición del ser."})

json_text = json.dumps(response, ensure_ascii=False, indent=2)
print(json_text)

# Append to log (gerard_log.txt) with cleaned text
clean_text = ''.join([item.get('content','') for item in response])
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log_entry = f"--- Conversación del {now} ---\nUsuario: LOCAL_SIMULADO\nPregunta: que es la eternidad\nRespuesta de GERARD: {clean_text}\n{'='*40}\n\n"
with open(os.path.join(ROOT, 'gerard_log.txt'), 'a', encoding='utf-8') as f:
    f.write(log_entry)

print('\n[Simulación guardada en gerard_log.txt]')

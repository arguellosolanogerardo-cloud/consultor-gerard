from datetime import datetime
import os
import sys

# Ensure repository root is on sys.path so we can import top-level modules
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import module (this will run module-level init; earlier imports were OK in this env)
import consultar_web

user_name = 'PRUEBA_USER'
now = datetime.now().strftime('%Y%m%d_%H%M%S')

sample_response_html = (
    f'<strong style="color:#28a745;">{user_name}:</strong> '
    "Este es un texto de prueba que contiene una cita (Fuente: ejemplo.srt, Timestamp: 00:00:10 --> 00:00:12) "
    "y además una parte enfatizada "
    "<span style=\"color:yellow; background-color: #333;\">IMPORTANTE</span>. "
    "Y un fragmento azul para la fuente: <span style=\"color:#87CEFA;\">(Fuente: otro.srt, Timestamp: 00:01:23 --> 00:01:25)</span>"
)

html_for_pdf = sample_response_html + f"<br/><br/><span style=\"color:#28a745;\">Usuario: {user_name}</span>"

pdf_bytes = consultar_web.generate_pdf_from_html(html_for_pdf, title_base=f"Q&A - {user_name}", user_name=user_name)

out_name = f"sample_QA_{user_name}_{now}.pdf"
out_path = os.path.join(os.getcwd(), out_name)
with open(out_path, 'wb') as f:
    f.write(pdf_bytes)

print(out_path)
print('size', os.path.getsize(out_path))

import os
import requests

key = os.environ.get('GOOGLE_API_KEY')
if not key:
    print('NO_KEY')
    raise SystemExit(2)

url = f'https://generativelanguage.googleapis.com/v1/models?key={key}'
try:
    r = requests.get(url, timeout=15)
    print('STATUS', r.status_code)
    print(r.text)
except Exception as e:
    print('ERR', e)
    raise SystemExit(3)

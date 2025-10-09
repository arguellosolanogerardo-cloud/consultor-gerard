import json
from types import SimpleNamespace

from consultar_web import (
    detect_gender_from_name,
    get_clean_text_from_json,
    format_docs_with_metadata,
)


def test_detect_gender_simple():
    assert detect_gender_from_name('María') == 'Femenino'
    assert detect_gender_from_name('Carlos') == 'Masculino'
    assert detect_gender_from_name('') == 'No especificar'
    assert detect_gender_from_name('X Æ A-12') == 'No especificar'


def test_get_clean_text_from_json_valid():
    arr = [
        {"type": "normal", "content": "Hola "},
        {"type": "emphasis", "content": "mundo"},
    ]
    s = json.dumps(arr)
    # add surrounding text to simulate model verbosity
    wrapped = f"Respuesta: {s} (fin)"
    assert get_clean_text_from_json(wrapped) == "Hola mundo"


def test_format_docs_with_metadata():
    doc = SimpleNamespace()
    doc.metadata = {"source": "[Spanish (auto-generated)] archivo.srt"}
    doc.page_content = "00:00:01,000 --> 00:00:03,000\nHola mundo"
    out = format_docs_with_metadata([doc])
    assert "Hola mundo" in out
    assert "archivo.srt" in out

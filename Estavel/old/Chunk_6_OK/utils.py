import logging
import fitz
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CONTEXT_CHARS

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def build_record(path: str, text: str) -> dict:
    try:
        doc = fitz.open(path)
        metadata = doc.metadata.copy() or {}
        metadata['numpages'] = doc.page_count
        doc.close()
    except Exception as e:
        logging.error(f"Erro ao extrair metadados: {e}")
        metadata = {}
    return {'text': text, 'info': metadata, 'version': "2.16.105"}

def is_valid_file(path: str) -> bool:
    from os import path as op
    if not op.isfile(path):
        logging.error(f"Arquivo não encontrado: {path}")
        return False
    if not path.lower().endswith(('.pdf', '.docx')):
        logging.error(f"Formato não suportado: {path}")
        return False
    return True

def generate_report(processed: List[str], errors: List[str]):
    logging.info(f"Processados: {len(processed)}")
    logging.warning(f"Erros: {len(errors)}")
    for e in errors:
        logging.warning(e)

def filter_first_pages(path: str, keep_from_page: int) -> str:
    """
    Abre o PDF em `path`, extrai texto de cada página e concatena
    a partir de `keep_from_page` (0-index).
    """
    doc = fitz.open(path)
    pages = []
    for idx in range(keep_from_page, doc.page_count):
        pages.append(doc.load_page(idx).get_text())
    doc.close()
    return "\n\n".join(pages)

def retrieve_with_padding(
    all_paragraphs: List[str],
    idx: int,
    pad: int
) -> (str, int, int):
    """
    Retorna contexto amplo centrado em `idx`:
    - inclui idx±pad parágrafos vizinhos;
    - limpa caracteres não-imprimíveis;
    - garante pelo menos metade do parágrafo central ou MIN_CONTEXT_CHARS.
    Retorna tupla: (contexto, low_index, high_index_exclusive).
    """
    total = len(all_paragraphs)
    low = max(0, idx - pad)
    high = min(total, idx + pad + 1)

    # limpa caracteres não-imprimíveis
    cleaned = ["".join(c for c in p if c.isprintable()) for p in all_paragraphs]

    context = "\n\n".join(cleaned[low:high])
    central_len = len(cleaned[idx])
    min_len = max(central_len // 2, MIN_CONTEXT_CHARS)

    # expande janela se contexto for curto
    while len(context) < min_len and (low > 0 or high < total):
        if low > 0:
            low -= 1
        if high < total:
            high += 1
        context = "\n\n".join(cleaned[low:high])

    return context, low, high

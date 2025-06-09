import logging
import os
from typing import List

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def build_record(path: str, text: str) -> dict:
    try:
        import fitz
        doc = fitz.open(path)
        info = doc.metadata.copy() or {}
        info['numpages'] = doc.page_count
        # conta imagens
        info['image_count'] = sum(len(page.get_images()) for page in doc)
        doc.close()
    except Exception as e:
        logging.error(f"Erro ao extrair metadados: {e}")
        info = {}
    return {'text': text, 'info': info, 'version': "2.16.105"}

def suppress_unrelated_warnings():
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("PyPDF2").setLevel(logging.ERROR)

def is_valid_file(path: str) -> bool:
    if not os.path.isfile(path):
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

def auto_chunk(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Divide o texto em pedaços de tamanho aproximado `chunk_size` palavras,
    com `overlap` palavras repetidas entre pedaços para manter contexto.
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

import logging
import os
from typing import List
import fitz
from PyPDF2 import PdfReader

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def build_record(path: str, text: str) -> dict:
    try:
        doc = fitz.open(path)
        info = doc.metadata.copy() or {}
        info['numpages'] = doc.page_count
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
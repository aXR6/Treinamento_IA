import logging
import os
from typing import List
import fitz
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def build_record(path: str, text: str) -> dict:
    """
    Extrai metadados do PDF com PyMuPDF.
    """
    try:
        doc = fitz.open(path)
        info = doc.metadata.copy() or {}
        info['numpages'] = doc.page_count
        doc.close()
    except Exception as e:
        logging.error(f"Erro ao extrair metadados: {e}")
        info = {}
    return {'text': text, 'info': info, 'version': "2.16.105"}

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

def chunk_text(text: str, metadata: dict) -> List[str]:
    """
    Divide texto em chunks automaticamente com padding dinâmico.
    - chunk_size calculado como max(CHUNK_SIZE, len(text)/(pages*2))
    - chunk_overlap = 10% do chunk_size
    """
    pages = metadata.get('numpages', 1)
    length = len(text)
    # prevê 2 chunks por página
    total_chunks = max(1, pages * 2)
    dynamic_chunk_size = max(CHUNK_SIZE, length // total_chunks)
    dynamic_overlap = max(CHUNK_OVERLAP, int(dynamic_chunk_size * 0.1))
    logging.info(
        f"Chunking automático: {total_chunks} janelas; tamanho={dynamic_chunk_size}; overlap={dynamic_overlap}"
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=dynamic_chunk_size,
        chunk_overlap=dynamic_overlap
    )
    return splitter.split_text(text)

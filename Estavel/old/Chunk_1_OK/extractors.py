import fitz
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import (
    PyPDFLoader, PDFMinerLoader, UnstructuredWordDocumentLoader
)
import logging

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

def is_extraction_allowed(path: str) -> bool:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            logging.warning(f"PDF criptografado: {path}")
            return False
        return True
    except Exception as e:
        logging.error(f"Erro ao verificar permissão: {e}")
        return False

def fallback_ocr(path: str, threshold: int = 100) -> str:
    try:
        doc = fitz.open(path)
        raw = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(raw.strip()) > threshold:
            return raw
        imgs = convert_from_path(path, dpi=300)
        return "\n\n".join(pytesseract.image_to_string(i, lang="eng+por") for i in imgs)
    except Exception as e:
        logging.error(f"Erro no OCR fallback: {e}")
        return ""

class PyPDFStrategy:
    def extract(self, path: str) -> str:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class PDFMinerStrategy:
    def extract(self, path: str) -> str:
        loader = PDFMinerLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class UnstructuredStrategy:
    def extract(self, path: str) -> str:
        loader = UnstructuredWordDocumentLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class OCRStrategy:
    def __init__(self, threshold: int = 100):
        self.threshold = threshold

    def extract(self, path: str) -> str:
        doc = fitz.open(path)
        raw = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(raw.strip()) > self.threshold:
            return raw
        imgs = convert_from_path(path, dpi=300)
        return "\n\n".join(pytesseract.image_to_string(i, lang="eng+por") for i in imgs)
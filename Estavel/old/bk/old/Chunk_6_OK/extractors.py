# extractors.py
import fitz
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader
)
from pdfminer.high_level import extract_text as pdfminer_extract_text
import logging

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
        images = convert_from_path(path, dpi=300)
        return "\n\n".join(
            pytesseract.image_to_string(img, lang="eng+por") for img in images
        )
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

class PDFMinerLowLevelStrategy:
    def extract(self, path: str) -> str:
        try:
            return pdfminer_extract_text(path)
        except Exception as e:
            logging.error(f"Erro no PDFMiner low-level: {e}")
            return ""

class UnstructuredStrategy:
    def extract(self, path: str) -> str:
        loader = UnstructuredWordDocumentLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class OCRStrategy:
    def __init__(self, threshold: int = 100):
        self.threshold = threshold

    def extract(self, path: str) -> str:
        try:
            doc = fitz.open(path)
            raw = "\n".join(page.get_text() for page in doc)
            doc.close()
            if len(raw.strip()) > self.threshold:
                return raw
            images = convert_from_path(path, dpi=300)
            return "\n\n".join(
                pytesseract.image_to_string(img, lang="eng+por") for img in images
            )
        except Exception as e:
            logging.error(f"Erro OCRStrategy: {e}")
            return ""

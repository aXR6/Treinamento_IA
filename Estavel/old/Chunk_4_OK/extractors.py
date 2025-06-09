# extractors.py

import logging
import fitz
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader
)
from pdfminer.high_level import extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def build_record(path: str, text: str) -> dict:
    """Extrai metadados e XMP com PyMuPDF e PyPDF2."""
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
    """Verifica se o PDF permite extração de texto."""
    try:
        reader = PdfReader(path)
        return not reader.is_encrypted
    except Exception as e:
        logging.error(f"Erro ao verificar permissão: {e}")
        return False

def fallback_ocr(path: str, threshold: int = 100) -> str:
    """Tenta extração digital; se insuficiente, usa OCR."""
    try:
        doc = fitz.open(path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text() or "")
        doc.close()
        raw_text = "\n".join(pages_text)
        if raw_text.strip() and len(raw_text.strip()) > threshold:
            logging.info("Texto extraído digitalmente via PyMuPDF.")
            return raw_text
        logging.info("Iniciando OCR...")
        images = convert_from_path(path, dpi=300)
        return "\n\n".join(pytesseract.image_to_string(img, lang="eng+por") for img in images)
    except Exception as e:
        logging.error(f"Erro no OCR fallback: {e}")
        return ""

# --- Estratégias de extração ---
class PyPDFStrategy:
    def extract(self, path: str) -> str:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class PDFMinerHighLevelStrategy:
    def extract(self, path: str) -> str:
        # usa a API high-level do pdfminer.six
        return extract_text(path)

class PDFMinerLowLevelStrategy:
    def extract(self, path: str) -> str:
        # pipeline customizado com LAParams e TextConverter
        output = StringIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            boxes_flow=None
        )
        device = TextConverter(rsrcmgr, output, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        with open(path, "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
        text = output.getvalue()
        device.close()
        output.close()
        return text

class UnstructuredStrategy:
    def extract(self, path: str) -> str:
        loader = UnstructuredWordDocumentLoader(path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

class OCRStrategy:
    def __init__(self, threshold: int = 100):
        self.threshold = threshold

    def extract(self, path: str) -> str:
        return fallback_ocr(path, self.threshold)
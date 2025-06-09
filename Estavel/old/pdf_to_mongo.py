#!/usr/bin/env python3
"""
Extrai conteúdo e metadados de PDF e grava no MongoDB
Menu aprimorado:
 1) Processar único PDF (OCR + metadata + storage)
 2) Processar todos os PDFs de uma pasta (mesmos passos)
 0) Sair
"""

import os
import sys
import json
import logging
import PyPDF2
from pymongo import MongoClient, errors as mongo_errors
from bson import Binary
from gridfs import GridFS
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF
from PyPDF2 import PdfReader


# Configurações
SCRIPT_VERSION = "2.16.105"
MONGO_URI      = "mongodb://app_user:senhaApp123@192.7.0.32:27017/ollama_chat?authSource=ollama_chat"
DB_NAME        = "ollama_chat"
COLL_PDF       = "PDF"
COLL_BIN       = "Arq_PDF"
GRIDFS_BUCKET  = "fs"

# ---------------------------------------------------------------------------
# Funções Auxiliares de Extração
# ---------------------------------------------------------------------------
def extract_text_ocr(path: str) -> str:
    """Extrai texto completo via OCR com pytesseract"""
    images = convert_from_path(path, dpi=300)
    texts = [pytesseract.image_to_string(img, lang="eng+por") for img in images]
    return "\n\n".join(texts)

# ---------------------------------------------------------------------------
# Construção de Registro Dinâmico
# ---------------------------------------------------------------------------
def build_pdf_record(path: str, text: str) -> dict:
    """Extrai metadados usando PyMuPDF e XMP via PyPDF2"""
    doc = fitz.open(path)
    info = doc.metadata.copy() or {}
    info['numpages'] = doc.page_count
    
    xmp = {}
    try:
        reader = PdfReader(path)
        if getattr(reader, 'xmp_metadata', None):
            xmp = dict(reader.xmp_metadata.custom_properties or {})
    except PyPDF2.errors.PdfReadError as e:
        logging.warning(f"Metadados XMP inválidos no arquivo {path}: {e}")
    
    return {'text': text, 'info': info, 'xmp_meta': xmp, 'version': SCRIPT_VERSION}

# ---------------------------------------------------------------------------
# Salvamentos no MongoDB
# ---------------------------------------------------------------------------
def save_metadata(record: dict) -> any:
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLL_PDF]
    try:
        res = col.insert_one(record)
        logging.info(f"Metadados salvos em '{COLL_PDF}', _id={res.inserted_id}")
        return res.inserted_id
    except mongo_errors.PyMongoError as e:
        logging.error(f"Erro ao salvar metadados: {e}")
        return None


def save_pdf_binary(name: str, path: str, parent_id=None) -> None:
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLL_BIN]
    try:
        data = open(path, 'rb').read()
        doc = {'filename': os.path.basename(path), 'file': Binary(data)}
        if name: doc['name'] = name
        if parent_id: doc['parent_id'] = parent_id
        res = col.insert_one(doc)
        logging.info(f"PDF binário '{name}' salvo em '{COLL_BIN}', _id={res.inserted_id}")
    except Exception as e:
        logging.error(f"Erro ao salvar PDF binário: {e}")


def save_gridfs(path: str, name: str) -> None:
    client = MongoClient(MONGO_URI)
    fs = GridFS(client[DB_NAME], collection=GRIDFS_BUCKET)
    try:
        fid = fs.put(open(path, 'rb'), filename=os.path.basename(path), metadata={'name': name})
        logging.info(f"PDF '{name}' salvo em GridFS, file_id={fid}")
    except Exception as e:
        logging.error(f"Erro em GridFS: {e}")

def extract_text_smart(path: str, threshold=100) -> str:
    """Extrai texto via PyMuPDF (se possível) ou OCR (se necessário)"""
    # Tentativa 1: Extrair texto diretamente via PyMuPDF
    doc = fitz.open(path)
    raw_text = ""
    for page in doc:
        raw_text += page.get_text()
    doc.close()

    # Critério: se o texto extraído tiver mais de X caracteres, usar diretamente
    if len(raw_text.strip()) > threshold:
        logging.info("Texto extraído diretamente (PDF digital).")
        return raw_text
    
    # Critério alternativo: se não houver texto, usar OCR
    logging.info("PDF escaneado detectado. Iniciando OCR...")
    images = convert_from_path(path, dpi=300)
    ocr_texts = [pytesseract.image_to_string(img, lang="eng+por") for img in images]
    return "\n\n".join(ocr_texts)

# ---------------------------------------------------------------------------
# Processamentos
# ---------------------------------------------------------------------------
def process_single_pdf():
    path = input("Caminho do PDF: ").strip()
    if not os.path.isfile(path): 
        logging.warning("PDF não encontrado.")
        return
    name = input("Nome do registro (ou vazio para usar nome do arquivo): ").strip() or None
    logging.info(f"Processando arquivo: {path}")
    
    # Decisão automática: texto direto ou OCR
    text = extract_text_smart(path)  # Aqui, o script decide sozinho!
    
    rec_id = save_metadata(build_pdf_record(path, text))
    save_pdf_binary(name or os.path.splitext(os.path.basename(path))[0], path, rec_id)
    save_gridfs(path, name or os.path.splitext(os.path.basename(path))[0])


def process_folder():
    folder = input("Caminho da pasta com PDFs: ").strip()
    if not os.path.isdir(folder): logging.warning("Pasta não encontrada."); return
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith('.pdf'):
            full = os.path.join(folder, fname)
            name = os.path.splitext(fname)[0]
            logging.info(f"\n--- Processando {fname} ---")
            text = extract_text_ocr(full)
            pid = save_metadata(build_pdf_record(full, text))
            save_pdf_binary(name, full, pid)
            save_gridfs(full, name)

# ---------------------------------------------------------------------------
# Menu Principal
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    while True:
        print("\n*** Menu Principal ***")
        print("1 - Processar um único PDF (OCR, metadados, binário, GridFS)")
        print("2 - Processar todos os PDFs de uma pasta")
        print("0 - Sair")
        choice = input("> ").strip()
        if choice == '1': process_single_pdf()
        elif choice == '2': process_folder()
        elif choice == '0': break
        else: logging.warning("Opção inválida.")
    logging.info("Encerrando aplicação.")

if __name__ == '__main__':
    main()
# main.py

import os
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from config import (
    MONGO_URI, DB_NAME, COLL_PDF, COLL_BIN, GRIDFS_BUCKET,
    OCR_THRESHOLD
)
from utils import setup_logging, is_valid_file, generate_report
from extractors import (
    is_extraction_allowed, fallback_ocr, build_record,
    PyPDFStrategy, PDFMinerHighLevelStrategy, PDFMinerLowLevelStrategy,
    UnstructuredStrategy, OCRStrategy
)
from storage import save_metadata, save_file_binary, save_gridfs
from pg_storage import save_to_postgres

# Suprime warnings irrelevantes
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
setup_logging()

STRATEGIES = {
    "pypdf":       PyPDFStrategy(),
    "pdfminer_hl": PDFMinerHighLevelStrategy(),
    "pdfminer_ll": PDFMinerLowLevelStrategy(),
    "unstructured":UnstructuredStrategy(),
    "ocr":         OCRStrategy(threshold=OCR_THRESHOLD),
}

def choose_database():
    return "postgres" if input("DB (1-mongo, 2-postgres) [1]: ").strip()=="2" else "mongo"

def choose_strategy():
    print("\n--- Método de Extração ---")
    print("1 - PyPDFLoader (LangChain)")
    print("2 - PDFMiner High-Level")
    print("3 - PDFMiner Low-Level")
    print("4 - Unstructured (.docx)")
    print("5 - OCR")
    m = input("Escolha [5]: ").strip()
    return {
        "1": "pypdf",
        "2": "pdfminer_hl",
        "3": "pdfminer_ll",
        "4": "unstructured",
        "5": "ocr"
    }.get(m, "ocr")

def process_file(path, strategy, db_choice, results):
    if not is_valid_file(path):
        results["errors"].append(path); return

    if not is_extraction_allowed(path):
        logging.warning(f"PDF criptografado/restrito, usando OCR: {path}")
        text = fallback_ocr(path, OCR_THRESHOLD)
    else:
        ext = os.path.splitext(path)[1].lower()
        key = "unstructured" if ext==".docx" else strategy
        text = STRATEGIES[key].extract(path)

    record = build_record(path, text)
    if db_choice=="mongo":
        pid = save_metadata(record, DB_NAME, COLL_PDF, MONGO_URI)
        if pid:
            save_file_binary(os.path.basename(path), path, pid, DB_NAME, COLL_BIN, MONGO_URI)
            save_gridfs(path, os.path.basename(path), DB_NAME, GRIDFS_BUCKET, MONGO_URI)
    else:
        save_to_postgres(os.path.basename(path), record["text"], record["info"])
    results["processed"].append(path)

def main():
    results = {"processed": [], "errors": []}
    while True:
        print("\n*** Menu Principal ***")
        print("1 - Processar único arquivo")
        print("2 - Processar pasta")
        print("0 - Sair")
        choice = input("> ").strip()
        if choice=='0': break

        db = choose_database()
        strat = choose_strategy()

        if choice=='1':
            p = input("Caminho do arquivo: ").strip()
            process_file(p, strat, db, results)
        else:
            folder = input("Caminho da pasta: ").strip()
            workers = int(input("Threads simultâneas [4]: ").strip() or 4)
            files = [f for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder,f))
                     and f.lower().endswith(('.pdf','.docx'))]
            print(f"Processando {len(files)} arquivos em {workers} threads...")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(tqdm(
                    ex.map(lambda f: process_file(os.path.join(folder,f), strat, db, results), files),
                    total=len(files), desc="Arquivos"
                ))
    generate_report(results["processed"], results["errors"])
    logging.info("Encerrando aplicação.")

if __name__=="__main__":
    main()
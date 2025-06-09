#!/usr/bin/env python3
import os
import logging
from concurrent.futures import ThreadPoolExecutor

from config import (
    MONGO_URI, DB_NAME, COLL_PDF, COLL_BIN, GRIDFS_BUCKET, OCR_THRESHOLD
)
from utils import setup_logging, is_valid_file, generate_report, build_record as build_meta
from extractors import (
    is_extraction_allowed,
    fallback_ocr,
    build_record,
    PyPDFStrategy,
    PDFMinerStrategy,
    PDFMinerLowLevelStrategy,
    UnstructuredStrategy,
    OCRStrategy
)
from storage import save_metadata, save_file_binary, save_gridfs
from pg_storage import save_to_postgres

# Supress warnings irrelevantes
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
setup_logging()

# Mapeia as estratégias disponíveis
STRATEGIES = {
    "pypdf":        PyPDFStrategy(),
    "pdfminer":     PDFMinerStrategy(),
    "pdfminer-low": PDFMinerLowLevelStrategy(),
    "unstructured": UnstructuredStrategy(),
    "ocr":          OCRStrategy(threshold=OCR_THRESHOLD),
}


def choose_database() -> str:
    choice = input("DB (1-mongo, 2-postgres) [1]: ").strip()
    return "postgres" if choice == "2" else "mongo"


def choose_strategy() -> str:
    print("\n--- Método de Extração ---")
    print("1 - PyPDFLoader (LangChain)")
    print("2 - PDFMinerLoader (LangChain)")
    print("3 - PDFMiner Low-Level (pdfminer.six)")
    print("4 - Unstructured (.docx)")
    print("5 - OCR")
    choice = input("Escolha [5]: ").strip()
    return {
        "1": "pypdf",
        "2": "pdfminer",
        "3": "pdfminer-low",
        "4": "unstructured",
        "5": "ocr"
    }.get(choice, "ocr")


def process_file(path: str, strategy: str, db_choice: str, results: dict):
    if not is_valid_file(path):
        results["errors"].append(path)
        return

    # Escolha OCR fallback se não permitido extrair
    if not is_extraction_allowed(path):
        text = fallback_ocr(path, OCR_THRESHOLD)
    else:
        ext = os.path.splitext(path)[1].lower()
        key = "unstructured" if ext == ".docx" else strategy
        text = STRATEGIES.get(key, STRATEGIES["ocr"]).extract(path)

    rec = build_meta(path, text)
    if db_choice == "mongo":
        pid = save_metadata(rec, DB_NAME, COLL_PDF, MONGO_URI)
        save_file_binary(os.path.basename(path), path, pid,
                         DB_NAME, COLL_BIN, MONGO_URI)
        save_gridfs(path, os.path.basename(path),
                    DB_NAME, GRIDFS_BUCKET, MONGO_URI)
    else:
        save_to_postgres(os.path.basename(path), rec["text"], rec["info"])

    results["processed"].append(path)


def main():
    results = {"processed": [], "errors": []}

    while True:
        print("\n*** Menu Principal ***")
        print("1 - Processar único arquivo")
        print("2 - Processar pasta")
        print("0 - Sair")
        choice = input("> ").strip()

        if choice == '0':
            break

        db = choose_database()
        strat = choose_strategy()

        if choice == '1':
            path = input("Caminho do arquivo: ").strip()
            process_file(path, strat, db, results)

        elif choice == '2':
            folder = input("Caminho da pasta: ").strip()
            with ThreadPoolExecutor() as executor:
                for fname in os.listdir(folder):
                    full = os.path.join(folder, fname)
                    if os.path.isfile(full) and fname.lower().endswith(('.pdf', '.docx')):
                        executor.submit(process_file, full, strat, db, results)

        else:
            logging.warning("Opção inválida.")

    generate_report(results["processed"], results["errors"])
    logging.info("Encerrando.")


if __name__ == "__main__":
    main()
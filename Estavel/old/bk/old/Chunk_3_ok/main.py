import os
import logging
from concurrent.futures import ThreadPoolExecutor

from config import (
    MONGO_URI, DB_NAME, COLL_PDF, COLL_BIN, GRIDFS_BUCKET, OCR_THRESHOLD
)
from utils import setup_logging, is_valid_file, generate_report, build_record
from extractors import (
    is_extraction_allowed, fallback_ocr, build_record as build_meta,
    PyPDFStrategy, PDFMinerStrategy, UnstructuredStrategy, OCRStrategy
)
from storage import save_metadata, save_file_binary, save_gridfs
from pg_storage import save_to_postgres

# Setup de logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
setup_logging()

STRATEGIES = {
    "pypdf":        PyPDFStrategy(),
    "pdfminer":     PDFMinerStrategy(),
    "unstructured": UnstructuredStrategy(),
    "ocr":          OCRStrategy(threshold=OCR_THRESHOLD),
}

def choose_database():
    c = input("DB (1-mongo, 2-postgres) [1]: ").strip()
    return "postgres" if c == "2" else "mongo"

def choose_strategy():
    c = input("Estratégia (1-pypdf, 2-pdfminer, 3-unstructured, 4-ocr) [4]: ").strip()
    return {"1":"pypdf","2":"pdfminer","3":"unstructured","4":"ocr"}.get(c, "ocr")

def process_file(path, strategy, db_choice, results):
    if not is_valid_file(path):
        results["errors"].append(path)
        return

    if not is_extraction_allowed(path):
        text = fallback_ocr(path, OCR_THRESHOLD)
    else:
        ext = os.path.splitext(path)[1].lower()
        key = "unstructured" if ext == ".docx" else strategy
        text = STRATEGIES.get(key, STRATEGIES["ocr"]).extract(path)

    rec = build_meta(path, text)
    filename = os.path.basename(path)

    if db_choice == "mongo":
        pid = save_metadata(rec, DB_NAME, COLL_PDF, MONGO_URI)
        save_file_binary(filename, path, pid, DB_NAME, COLL_BIN, MONGO_URI)
        save_gridfs(path, filename, DB_NAME, GRIDFS_BUCKET, MONGO_URI)
    else:
        save_to_postgres(filename, rec["text"], rec["info"])

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
            with ThreadPoolExecutor() as ex:
                for f in os.listdir(folder):
                    full = os.path.join(folder, f)
                    if os.path.isfile(full) and f.lower().endswith(('.pdf','.docx')):
                        ex.submit(process_file, full, strat, db, results)
        else:
            logging.warning("Opção inválida.")

    generate_report(results["processed"], results["errors"])
    logging.info("Encerrando.")

if __name__ == "__main__":
    main()
#config.py

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# — Embeddings Ollama ——————————————————————————
OLLAMA_API_URL         = os.getenv("OLLAMA_API_URL")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")
EMBEDDING_DIM          = int(os.getenv("EMBEDDING_DIM", "1024"))

# — Chunking adaptativo ——————————————————————————
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# — RAG por parágrafo —————————————————————————————————
PARAGRAPH_CHUNK_SIZE    = int(os.getenv("PARAGRAPH_CHUNK_SIZE", "10000"))
PARAGRAPH_OVERLAP       = int(os.getenv("PARAGRAPH_OVERLAP", "0"))

# — Agrupamento de parágrafos ————————————————————
PARAGRAPH_GROUP_SIZE    = int(os.getenv("PARAGRAPH_GROUP_SIZE", "3"))
PARAGRAPH_GROUP_OVERLAP = int(os.getenv("PARAGRAPH_GROUP_OVERLAP", "1"))

# — Filtro de primeiras páginas —————————————————————
KEEP_FROM_PAGE = int(os.getenv("KEEP_FROM_PAGE", "2"))

# — Padding de retrieval (vizinhança) ———————————————
RETRIEVAL_PADDING_PARAGRAPHS = int(os.getenv("RETRIEVAL_PADDING_PARAGRAPHS", "1"))

# — OCR threshold ———————————————————————————————————
OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "100"))

# — MongoDB ———————————————————————————————————————
MONGO_URI     = os.getenv("MONGO_URI")
DB_NAME       = os.getenv("DB_NAME", "ollama_chat")
COLL_PDF      = os.getenv("COLL_PDF", "PDF_")
COLL_BIN      = os.getenv("COLL_BIN", "Arq_PDF")
GRIDFS_BUCKET = os.getenv("GRIDFS_BUCKET", "fs")

# — PostgreSQL —————————————————————————————————————
PG_HOST       = os.getenv("PG_HOST")
PG_PORT       = os.getenv("PG_PORT")
PG_DB         = os.getenv("PG_DB")
PG_USER       = os.getenv("PG_USER")
PG_PASSWORD   = os.getenv("PG_PASSWORD")
PG_SCHEMA     = os.getenv("PG_SCHEMA", "public")

# — CSV locais (NVD) ——————————————————————————————
CSV_FULL      = os.getenv("CSV_FULL", "vulnerabilidades_full.csv")
CSV_INCR      = os.getenv("CSV_INCR", "vulnerabilidades_incrementais.csv")

# — Anos completos (2002 até o ano atual) —————————————
YEARS = list(range(2002, datetime.utcnow().year + 1))

# — Mínimo de caracteres de contexto (50% do parágrafo ou este valor) —
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS", "500"))

def validate_config():
    """Validações básicas das configurações."""
    checks = [
        ("EMBEDDING_DIM", EMBEDDING_DIM),
        ("CHUNK_SIZE", CHUNK_SIZE),
        ("CHUNK_OVERLAP", CHUNK_OVERLAP),
        ("PARAGRAPH_CHUNK_SIZE", PARAGRAPH_CHUNK_SIZE),
        ("PARAGRAPH_OVERLAP", PARAGRAPH_OVERLAP),
        ("PARAGRAPH_GROUP_SIZE", PARAGRAPH_GROUP_SIZE),
        ("PARAGRAPH_GROUP_OVERLAP", PARAGRAPH_GROUP_OVERLAP),
        ("KEEP_FROM_PAGE", KEEP_FROM_PAGE),
        ("RETRIEVAL_PADDING_PARAGRAPHS", RETRIEVAL_PADDING_PARAGRAPHS),
        ("OCR_THRESHOLD", OCR_THRESHOLD),
        ("MIN_CONTEXT_CHARS", MIN_CONTEXT_CHARS),
    ]
    for name, val in checks:
        if val is None or (isinstance(val, int) and val < 0):
            logging.error(f"{name} inválido ({val}); deve ser >= 0.")
    if not (MONGO_URI and PG_HOST and PG_USER and PG_PASSWORD):
        logging.warning("Algumas variáveis críticas não estão definidas.")
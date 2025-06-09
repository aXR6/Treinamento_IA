import os
from dotenv import load_dotenv
import logging

load_dotenv()

# API de embeddings
OLLAMA_API_URL        = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/embeddings")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")

# MongoDB
MONGO_URI      = os.getenv("MONGO_URI")
DB_NAME        = os.getenv("DB_NAME", "ollama_chat")
COLL_PDF       = os.getenv("COLL_PDF", "PDF_")
COLL_BIN       = os.getenv("COLL_BIN", "Arq_PDF")
GRIDFS_BUCKET  = os.getenv("GRIDFS_BUCKET", "fs")

# PostgreSQL
PG_HOST        = os.getenv("PG_HOST")
PG_PORT        = os.getenv("PG_PORT")
PG_DB          = os.getenv("PG_DB")
PG_USER        = os.getenv("PG_USER")
PG_PASSWORD    = os.getenv("PG_PASSWORD")
PG_SCHEMA      = os.getenv("PG_SCHEMA", "public")

# Outros parâmetros
OCR_THRESHOLD  = int(os.getenv("OCR_THRESHOLD", 100))
EMBEDDING_DIM  = int(os.getenv("EMBEDDING_DIM", 1024))
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 200))

def validate_config():
    if not MONGO_URI or not PG_HOST:
        logging.warning("Variáveis críticas não definidas.")
    for name, val in [("EMBEDDING_DIM", EMBEDDING_DIM), ("CHUNK_SIZE", CHUNK_SIZE), ("CHUNK_OVERLAP", CHUNK_OVERLAP)]:
        if val <= 0:
            logging.error(f"{name} inválido ({val}); deve ser > 0.")

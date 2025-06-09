# config.py

import os
from dotenv import load_dotenv
import logging

load_dotenv()

# URL da API de embeddings do Ollama
OLLAMA_API_URL = os.getenv(
    "OLLAMA_API_URL",
    "http://localhost:11434/api/embeddings"
)

# Modelo de embeddings do Ollama
OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "mxbai-embed-large"
)

# MongoDB
MONGO_URI     = os.getenv("MONGO_URI")
DB_NAME       = os.getenv("DB_NAME", "ollama_chat")
COLL_PDF      = os.getenv("COLL_PDF", "PDF_")
COLL_BIN      = os.getenv("COLL_BIN", "Arq_PDF")
GRIDFS_BUCKET = os.getenv("GRIDFS_BUCKET", "fs")

# PostgreSQL
PG_HOST       = os.getenv("PG_HOST")
PG_PORT       = os.getenv("PG_PORT")
PG_DB         = os.getenv("PG_DB")
PG_USER       = os.getenv("PG_USER")
PG_PASSWORD   = os.getenv("PG_PASSWORD")
PG_SCHEMA     = os.getenv("PG_SCHEMA", "public")

# Outros parâmetros
OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", 100))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))

def validate_config():
    if not MONGO_URI:
        logging.warning("MONGO_URI não definido. Usando valor padrão.")
    if not PG_HOST:
        logging.warning("PG_HOST não definido. Verifique suas variáveis de ambiente.")
    if EMBEDDING_DIM <= 0:
        logging.error("EMBEDDING_DIM inválido. Deve ser > 0.")
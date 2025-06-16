#config.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente exclusivamente do arquivo `.env` na raiz do
# projeto. `override=True` garante que valores definidos nesse arquivo
# substituam variáveis já presentes no ambiente.
load_dotenv(Path(__file__).resolve().with_name('.env'), override=True)

# Valor padrao seguro para o comprimento maximo de sequencias
DEFAULT_MAX_SEQ_LENGTH = 128

# — PostgreSQL Connection
PG_HOST     = os.getenv("PG_HOST")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_USER     = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB_PDF   = os.getenv("PG_DB_PDF")
PG_DB_QA    = os.getenv("PG_DB_QA")
PG_DB_CVE   = os.getenv("PG_DB_CVE")

# — Modelos de Embedding & Chunking
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL")
SERAFIM_EMBEDDING_MODEL = os.getenv("SERAFIM_EMBEDDING_MODEL")
MINILM_L6_V2            = os.getenv("MINILM_L6_V2")
MINILM_L12_V2           = os.getenv("MINILM_L12_V2")
MPNET_EMBEDDING_MODEL   = os.getenv("MPNET_EMBEDDING_MODEL")
SBERT_MODEL_NAME        = os.getenv("SBERT_MODEL_NAME")
# Modelos para geração de perguntas e respostas
QG_MODEL = os.getenv("QG_MODEL", "valhalla/t5-base-qa-qg-hl")
QA_MODEL = os.getenv("QA_MODEL", QG_MODEL)
# Ativa geracao de respostas via prompt explicito "question: ... context: ...".
QA_EXPLICIT_PROMPT = os.getenv("QA_EXPLICIT_PROMPT", "0").lower() in ("1", "true", "yes")
TRAINING_MODEL_NAME    = os.getenv(
    "TRAINING_MODEL_NAME",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
)

# — Dimensões de embedding
DIM_MXBAI     = int(os.getenv("DIM_MXBAI", "0"))
DIM_SERAFIM   = int(os.getenv("DIM_SERAFIM", "0"))
DIM_MINILM_L6 = int(os.getenv("DIM_MINILM_L6", "0"))
DIM_MINIL12   = int(os.getenv("DIM_MINIL12", "0"))
DIM_MPNET     = int(os.getenv("DIM_MPNET", "0"))

# — Parâmetros OCR
OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "100"))
# Usa 'eng+por' como padrão se OCR_LANGUAGES não estiver definido
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+por")
# Timeout máximo para pdf2image.convert_from_path (segundos)
PDF2IMAGE_TIMEOUT = int(os.getenv("PDF2IMAGE_TIMEOUT", "600"))
# Parâmetros adicionais passados ao Tesseract (ex: "--oem 3 --psm 6")
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "")

# — Parâmetros de chunking
CHUNK_SIZE                   = int(os.getenv("CHUNK_SIZE", "0"))
CHUNK_OVERLAP                = int(os.getenv("CHUNK_OVERLAP", "0"))
SLIDING_WINDOW_OVERLAP_RATIO = float(os.getenv("SLIDING_WINDOW_OVERLAP_RATIO", "0.0"))

# Comprimento maximo de sequencia para pipelines de QA.
# Se nao especificado ou for muito pequeno, usa DEFAULT_MAX_SEQ_LENGTH.
try:
    MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH)))
except ValueError:
    MAX_SEQ_LENGTH = DEFAULT_MAX_SEQ_LENGTH
if MAX_SEQ_LENGTH < 8:
    logging.warning(
        "MAX_SEQ_LENGTH=%s invalido; usando %s", MAX_SEQ_LENGTH, DEFAULT_MAX_SEQ_LENGTH
    )
    MAX_SEQ_LENGTH = DEFAULT_MAX_SEQ_LENGTH

SEPARATORS = os.getenv("SEPARATORS", "").split("|")

# --- Parâmetros de treinamento
EVAL_STEPS       = int(os.getenv("EVAL_STEPS", "500"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.1"))
# Hiperpar\u00e2metros adicionais para o fine-tuning
LEARNING_RATE               = float(os.getenv("LEARNING_RATE", "5e-5"))
WEIGHT_DECAY                = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_STEPS                = int(os.getenv("WARMUP_STEPS", "100"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
LR_SCHEDULER_TYPE           = os.getenv("LR_SCHEDULER_TYPE", "linear")
# Quantos processos paralelos usar ao tokenizar datasets
TOKENIZE_NUM_PROC           = int(os.getenv("TOKENIZE_NUM_PROC", "1"))
# Workers do PyTorch DataLoader usados pelo Trainer
DATALOADER_NUM_WORKERS      = int(os.getenv("DATALOADER_NUM_WORKERS", "0"))

def validate_config():
    missing = []
    for var in ("PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD", "PG_DB_PDF", "PG_DB_QA", "PG_DB_CVE"):
        if not globals().get(var):
            missing.append(var)
    if missing:
        logging.error(f"Variáveis críticas faltando: {missing}")
        raise RuntimeError(f"Variáveis faltando: {missing}")

import logging
import json
import requests
from psycopg2 import pool, sql
import psycopg2
from tqdm import tqdm
from utils import retrieve_with_padding
from config import (
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA,
    OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM,
    PARAGRAPH_CHUNK_SIZE, PARAGRAPH_OVERLAP,
    RETRIEVAL_PADDING_PARAGRAPHS
)
from new_paragraph_splitter import ParagraphSplitter

# Inicializa pool de conexões PostgreSQL
POSTGRES_POOL = pool.ThreadedConnectionPool(
    minconn=1, maxconn=20,
    host=PG_HOST, port=PG_PORT,
    dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
)
logging.info("Pool PostgreSQL inicializado (1-20 conexões)")

def get_connection():
    return POSTGRES_POOL.getconn()

def release_connection(conn):
    POSTGRES_POOL.putconn(conn)

def generate_embedding(text: str) -> list[float]:
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        emb = resp.json().get("embedding", [])
        if len(emb) != EMBEDDING_DIM:
            emb = emb[:EMBEDDING_DIM] if len(emb) > EMBEDDING_DIM else emb + [0.0] * (EMBEDDING_DIM - len(emb))
        return emb
    except Exception as e:
        logging.error(f"Falha gerando embedding: {e}")
        return [0.0] * EMBEDDING_DIM

def save_to_postgres(filename: str, text: str, metadata: dict):
    conn = get_connection()
    try:
        # 1) Quebra o texto em parágrafos
        splitter = ParagraphSplitter(PARAGRAPH_CHUNK_SIZE, PARAGRAPH_OVERLAP)
        paragraphs = splitter.split(text)
        total = len(paragraphs)
        logging.info(f"'{filename}' dividido em {total} parágrafos para contexto amplo.")

        with conn.cursor() as cur:
            for idx in tqdm(range(total), desc=filename, unit="parágrafo"):
                # 2) Recupera contexto amplo + índices
                chunk, low, high = retrieve_with_padding(
                    all_paragraphs=paragraphs,
                    idx=idx,
                    pad=RETRIEVAL_PADDING_PARAGRAPHS
                )
                # 3) Gera embedding
                emb = generate_embedding(chunk)
                # 4) Prepara metadados com intervalo real
                record = {
                    **metadata,
                    "__parent_file": filename,
                    "__start_para_idx": low,
                    "__end_para_idx": high - 1
                }
                # 5) Persiste no banco
                cur.execute(
                    sql.SQL("INSERT INTO {}.documents (content, metadata, embedding) VALUES (%s, %s, %s)")
                        .format(sql.Identifier(PG_SCHEMA)),
                    (chunk, json.dumps(record, ensure_ascii=True), emb)
                )

        conn.commit()
        logging.info(f"'{filename}' salvo com contexto amplo em cada parágrafo.")
    except psycopg2.Error as e:
        logging.error(f"Erro no insert Postgres: {e}")
    finally:
        release_connection(conn)

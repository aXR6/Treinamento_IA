# pg_storage.py

import logging
import psycopg2
from psycopg2 import sql
import requests
import json

from config import (
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA,
    OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM
)

def generate_embedding(text: str) -> list[float]:
    """Gera embedding e ajusta dimensão por pad/truncate."""
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        emb = resp.json().get("embedding", [])
        # pad ou truncate
        if len(emb) != EMBEDDING_DIM:
            emb = emb[:EMBEDDING_DIM] + [0.0] * max(0, EMBEDDING_DIM - len(emb))
        return emb
    except Exception as e:
        logging.error(f"Falha gerando embedding: {e}")
        return [0.0] * EMBEDDING_DIM

def connect_postgres():
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
        )
        logging.info("Conectado ao PostgreSQL")
        return conn
    except Exception as e:
        logging.error(f"Erro conexão PostgreSQL: {e}")
        return None

def save_to_postgres(filename: str, chunks: list[dict], doc_metadata: dict):
    """
    chunks: lista de dicts com keys:
      - 'index': número do chunk
      - 'text': conteúdo do chunk
      - 'metrics': dict com métricas calculadas
    doc_metadata: metadados do documento original
    """
    conn = connect_postgres()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            for chunk in chunks:
                content = chunk["text"]
                # inclui o índice e metadados do doc original
                metadata = {
                    "source_file": filename,
                    "chunk_index": chunk["index"],
                    "chunk_metrics": chunk["metrics"],
                    **doc_metadata
                }
                embedding = generate_embedding(content)
                cur.execute(
                    sql.SQL("INSERT INTO {}.documents (content, metadata, embedding) VALUES (%s, %s, %s)")
                       .format(sql.Identifier(PG_SCHEMA)),
                    (content, json.dumps(metadata), embedding)
                )
        conn.commit()
        logging.info(f"'{filename}' salvo no PostgreSQL ({len(chunks)} chunks).")
    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error: {e}")
    finally:
        conn.close()
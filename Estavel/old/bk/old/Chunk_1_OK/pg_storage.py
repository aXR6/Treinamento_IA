import logging
import psycopg2
from psycopg2 import sql
import requests
import json

from config import (
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA,
    OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM
)
from utils import chunk_text

def generate_embedding(text: str) -> list[float]:
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding", [])
        orig = len(embedding)
        if orig != EMBEDDING_DIM:
            logging.warning(f"Embedding com {orig} dims; ajustando para {EMBEDDING_DIM}.")
            if orig > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            else:
                embedding += [0.0] * (EMBEDDING_DIM - orig)
        return embedding
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

def save_to_postgres(filename: str, text: str, metadata: dict):
    conn = connect_postgres()
    if not conn:
        return

    try:
        # 1) Gerar chunks automaticamente
        chunks = chunk_text(text, metadata)

        # 2) Para cada chunk, gerar embedding e salvar
        with conn.cursor() as cur:
            for idx, chunk in enumerate(chunks):
                emb = generate_embedding(chunk)
                data = {
                    **metadata,
                    "__parent_file": filename,
                    "__chunk_index": idx
                }
                cur.execute(
                    sql.SQL("INSERT INTO {}.documents (content, metadata, embedding) VALUES (%s, %s, %s)")
                       .format(sql.Identifier(PG_SCHEMA)),
                    (chunk, json.dumps(data), emb)
                )
        conn.commit()
        logging.info(f"'{filename}' salvo como {len(chunks)} chunks no PostgreSQL.")
    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error: {e}")
    finally:
        conn.close()
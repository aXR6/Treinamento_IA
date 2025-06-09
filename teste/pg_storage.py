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
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        orig = len(embedding)
        if orig != EMBEDDING_DIM:
            logging.warning(
                f"Embedding com {orig} dims; ajustando para {EMBEDDING_DIM}."
            )
            if orig > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            else:
                embedding = embedding + [0.0] * (EMBEDDING_DIM - orig)
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
        embedding = generate_embedding(text)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("INSERT INTO {}.documents (content, metadata, embedding) VALUES (%s, %s, %s)")
                   .format(sql.Identifier(PG_SCHEMA)),
                (text, json.dumps(metadata), embedding)
            )
            conn.commit()
            logging.info(f"'{filename}' salvo no PostgreSQL.")
    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error: {e}")  # :contentReference[oaicite:1]{index=1}
    finally:
        conn.close()
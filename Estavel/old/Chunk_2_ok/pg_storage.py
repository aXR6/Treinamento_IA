import logging
import psycopg2
from psycopg2 import sql
import requests
import json

from config import (
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA,
    OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM
)
from utils import auto_chunk

def generate_embedding(text: str) -> list[float]:
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        # ajusta pad/truncate
        orig = len(embedding)
        if orig != EMBEDDING_DIM:
            logging.warning(f"Embedding com {orig} dims; ajustando para {EMBEDDING_DIM}.")
            embedding = (embedding[:EMBEDDING_DIM]
                         if orig > EMBEDDING_DIM
                         else embedding + [0.0] * (EMBEDDING_DIM - orig))
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
    """
    Divide o texto em chunks, gera embeddings e salva cada chunk
    como uma linha na tabela `documents` com colunas:
      content, metadata, embedding
    Onde `metadata` recebe também o nome de arquivo e índice do chunk.
    """
    conn = connect_postgres()
    if not conn:
        return

    try:
        chunks = auto_chunk(text)
        with conn.cursor() as cur:
            for idx, chunk in enumerate(chunks, start=1):
                emb = generate_embedding(chunk)
                # anexa filename e índice aos metadados JSON
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "filename": filename,
                    "chunk_index": idx
                })
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {}.documents (content, metadata, embedding) "
                        "VALUES (%s, %s, %s)"
                    ).format(sql.Identifier(PG_SCHEMA)),
                    (chunk, json.dumps(chunk_meta), emb)
                )
            conn.commit()
        logging.info(f"'{filename}' salvo no PostgreSQL ({len(chunks)} chunks).")
    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error: {e}")
    finally:
        conn.close()
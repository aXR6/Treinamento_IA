import logging
import json
import requests
import psycopg2
from psycopg2 import pool, sql
from tqdm import tqdm

from config import (
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA,
    OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM
)
from adaptive_chunker import adaptive_chunk

# --- Inicialização do pool de conexões ---
try:
    POSTGRES_POOL = pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=20,
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )
    logging.info("PostgreSQL connection pool initialized (1-20 connections)")
except Exception as e:
    logging.error(f"Failed to initialize Postgres pool: {e}")
    raise


def get_connection():
    """Obtém uma conexão do pool sem criar nova conexão a cada chamada."""
    return POSTGRES_POOL.getconn()


def release_connection(conn):
    """Devolve a conexão ao pool para reutilização."""
    POSTGRES_POOL.putconn(conn)


def generate_embedding(text: str) -> list[float]:
    """Chama a API de embeddings e ajusta a dimensão por pad/truncate."""
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
            logging.warning(
                f"Embedding com {orig} dims; ajustando para {EMBEDDING_DIM}."
            )
            if orig > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            else:
                embedding += [0.0] * (EMBEDDING_DIM - orig)
        return embedding
    except Exception as e:
        logging.error(f"Falha gerando embedding: {e}")
        return [0.0] * EMBEDDING_DIM


def save_to_postgres(filename: str, text: str, metadata: dict):
    """
    Chunkiza semanticamente, gera embedding e persiste cada chunk.
    Exibe progresso individual com tqdm.
    """
    conn = get_connection()
    if not conn:
        logging.error("Não foi possível obter conexão do pool")
        return

    try:
        chunks = adaptive_chunk(text, metadata)
        total = len(chunks)
        logging.info(f"Processando '{filename}' → {total} chunks")
        with conn.cursor() as cur:
            for idx, chunk in enumerate(tqdm(chunks, desc=filename, unit='chunk')):
                emb = generate_embedding(chunk)
                data = {
                    **metadata,
                    "__parent_file": filename,
                    "__chunk_index": idx
                }
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {}.documents (content, metadata, embedding) "
                        "VALUES (%s, %s, %s)"
                    ).format(sql.Identifier(PG_SCHEMA)),
                    (chunk, json.dumps(data), emb)
                )
        conn.commit()
        logging.info(f"'{filename}' salvo como {total} chunks no PostgreSQL.")
    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error durante insert: {e}")
    finally:
        release_connection(conn)
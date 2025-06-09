import logging, json, requests, psycopg2
from psycopg2 import pool, sql
from tqdm import tqdm
from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA, OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM
from adaptive_chunker import hierarchical_chunk

# Inicializa pool de conexões
POSTGRES_POOL = pool.ThreadedConnectionPool(
    1, 20,
    host=PG_HOST, port=PG_PORT,
    dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
)

def get_connection(): return POSTGRES_POOL.getconn()
def release_connection(conn): POSTGRES_POOL.putconn(conn)

def generate_embedding(text: str) -> list[float]:
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={'model': OLLAMA_EMBEDDING_MODEL, 'prompt': text},
            timeout=60
        )
        resp.raise_for_status()
        emb = resp.json().get('embedding', [])
        if len(emb) != EMBEDDING_DIM:
            logging.warning(f"dims {len(emb)}→{EMBEDDING_DIM}")
            emb = (emb[:EMBEDDING_DIM] if len(emb)>EMBEDDING_DIM else emb + [0.0]*(EMBEDDING_DIM-len(emb)))
        return emb
    except Exception as e:
        logging.error(f"Embedding fail: {e}")
        return [0.0]*EMBEDDING_DIM


def save_to_postgres(filename: str, text: str, metadata: dict):
    conn = get_connection()
    if not conn:
        logging.error('Sem conexão')
        return
    try:
        chunks = hierarchical_chunk(text, metadata)
        logging.info(f"Processando '{filename}' → {len(chunks)} chunks")
        with conn.cursor() as cur:
            for idx, chunk in enumerate(tqdm(chunks, desc=filename, unit='chunk')):
                emb = generate_embedding(chunk)
                data = {**metadata, '__parent': filename, '__chunk_index': idx}
                cur.execute(
                    sql.SQL('INSERT INTO {}.documents (content, metadata, embedding) VALUES (%s, %s, %s)')
                         .format(sql.Identifier(PG_SCHEMA)),
                    (chunk, json.dumps(data), emb)
                )
        conn.commit()
        logging.info(f"Salvo {len(chunks)} chunks de '{filename}'")
    except psycopg2.Error as e:
        logging.error(f"Postgres error: {e}")
    finally:
        release_connection(conn)
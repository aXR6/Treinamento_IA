import logging, json, requests, psycopg2
from psycopg2 import pool, sql
from tqdm import tqdm
from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA, OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM
from utils import chunk_text

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
        resp = requests.post(OLLAMA_API_URL, json={'model': OLLAMA_EMBEDDING_MODEL, 'prompt': text}, timeout=60)
        resp.raise_for_status()
        emb = resp.json().get('embedding', [])
        if len(emb) != EMBEDDING_DIM:
            logging.warning('dims {}→{}'.format(len(emb), EMBEDDING_DIM))
            emb = (emb[:EMBEDDING_DIM] if len(emb)>EMBEDDING_DIM else emb + [0.0]*(EMBEDDING_DIM-len(emb)))
        return emb
    except Exception as e:
        logging.error('Embedding fail: {}'.format(e))
        return [0.0]*EMBEDDING_DIM

def save_to_postgres(filename: str, text: str, metadata: dict):
    conn = get_connection()
    if not conn:
        logging.error('Sem conexão')
        return
    try:
        chunks = chunk_text(text, metadata)
        with conn.cursor() as cur:
            for i, chunk in enumerate(tqdm(chunks, desc=filename)):
                emb = generate_embedding(chunk)
                meta = {**metadata, '__parent': filename, '__idx': i, '__preview': chunk[:50]+'...'}
                cur.execute(
                    sql.SQL('INSERT INTO {}.documents (content, metadata, embedding) VALUES (%s,%s,%s)')
                         .format(sql.Identifier(PG_SCHEMA)),
                    (chunk, json.dumps(meta), emb)
                )
        conn.commit()
        logging.info('Salvo {} chunks de {}'.format(len(chunks), filename))
    except Exception as e:
        logging.error('Postgres error: {}'.format(e))
    finally:
        release_connection(conn)
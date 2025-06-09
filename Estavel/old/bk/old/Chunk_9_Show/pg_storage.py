#pg_storage.py
import logging, json, requests, psycopg2
from psycopg2 import pool, sql
from tqdm import tqdm
from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA, OLLAMA_API_URL, OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIM
from adaptive_chunker import hierarchical_chunk

# Initialize a thread-safe connection pool
POSTGRES_POOL = pool.ThreadedConnectionPool(
    1, 20,
    host=PG_HOST, port=PG_PORT,
    dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
)

def get_connection():
    return POSTGRES_POOL.getconn()

def release_connection(conn):
    POSTGRES_POOL.putconn(conn)


def generate_embedding(text: str) -> list[float]:
    """
    Calls the Ollama embedding API, ensures the returned embedding matches EMBEDDING_DIM,
    padding or truncating with zeros if necessary.
    """
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={'model': OLLAMA_EMBEDDING_MODEL, 'prompt': text},
            timeout=60
        )
        resp.raise_for_status()
        emb = resp.json().get('embedding', [])
        if len(emb) != EMBEDDING_DIM:
            logging.warning(f"Embedding dimension mismatch: got {len(emb)}, expected {EMBEDDING_DIM}")
            # Pad or truncate
            emb = (emb[:EMBEDDING_DIM] if len(emb) > EMBEDDING_DIM
                   else emb + [0.0] * (EMBEDDING_DIM - len(emb)))
        return emb
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return [0.0] * EMBEDDING_DIM


def save_to_postgres(filename: str, text: str, metadata: dict):
    """
    Splits the text into chunks, generates embeddings, and inserts each chunk
    along with metadata into PostgreSQL. Sanitizes null characters before insertion.
    """
    conn = get_connection()
    if not conn:
        logging.error('No PostgreSQL connection available')
        return
    try:
        # Create chunks via hierarchical chunking
        chunks = hierarchical_chunk(text, metadata)
        logging.info(f"Processing '{filename}' → {len(chunks)} chunks")
        with conn.cursor() as cur:
            for idx, chunk in enumerate(tqdm(chunks, desc=filename, unit='chunk')):
                # Remove NUL characters to avoid ValueError
                clean_chunk = chunk.replace('\x00', '')
                emb = generate_embedding(clean_chunk)
                # Prepare metadata with parent and chunk index
                record = {**metadata, '__parent': filename, '__chunk_index': idx}
                # Serialize and sanitize metadata JSON
                meta_json = json.dumps(record).replace('\x00', '')
                # Insert into PostgreSQL
                cur.execute(
                    sql.SQL('INSERT INTO {}.documents (content, metadata, embedding) VALUES (%s, %s, %s)')
                         .format(sql.Identifier(PG_SCHEMA)),
                    (clean_chunk, meta_json, emb)
                )
        conn.commit()
        logging.info(f"Saved {len(chunks)} chunks for '{filename}'")
    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error inserting data: {e}")
    finally:
        release_connection(conn)
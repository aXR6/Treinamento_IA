import logging
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Carrega modelo SBERT para embeddings de sentenças
# Usa o caminho completo do repositório Hugging Face
SEM_MODEL = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

from config import CHUNK_SIZE, CHUNK_OVERLAP


def semantic_chunk(text: str, num_chunks: int) -> list[str]:
    """
    Chunk semântico: agrupa sentenças via clustering de embeddings SBERT.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return []
    # Gera embeddings das sentenças
    embs = SEM_MODEL.encode(sentences)
    # Número de clusters não pode exceder número de sentenças
    n = min(num_chunks, len(sentences))
    kmeans = KMeans(n_clusters=n)
    labels = kmeans.fit_predict(embs)
    chunks = {i: [] for i in range(n)}
    for sent, lbl in zip(sentences, labels):
        chunks[lbl].append(sent)
    # Reconstrói texto de cada cluster
    return ['. '.join(v) for v in chunks.values()]


def adaptive_chunk(text: str, metadata: dict) -> list[str]:
    """
    Seleciona estratégia de chunking conforme tipo de documento:
    - .docx ou PDF escaneado: splitter recursivo com padding.
    - PDF digital: chunking semântico em 2x número de páginas.
    """
    is_docx = metadata.get('format', '').lower().endswith('docx')
    is_scan = metadata.get('is_encrypted', False)
    pages = metadata.get('numpages', 1)

    # Heurístico recursivo para docx e scans
    if is_docx or is_scan:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return splitter.split_text(text)

    # Chunk semântico para PDF digital
    return semantic_chunk(text, pages * 2)

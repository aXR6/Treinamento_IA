import logging
import fitz
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from config import CHUNK_SIZE, CHUNK_OVERLAP

def is_complex_layout(metadata: dict) -> bool:
    # Exemplo: layout complexo se houver muitas imagens ou colunas
    return metadata.get('numpages', 1) < 5 and metadata.get('HasImages', False)

def is_long_narrative(metadata: dict) -> bool:
    # Exemplo: narrativa longa para > 50 páginas
    return metadata.get('numpages', 0) > 50

def structural_chunk(text: str, metadata: dict) -> List[str]:
    """
    Divide por página usando PyMuPDF, depois aplica RecursiveCharacterTextSplitter.
    """
    doc = fitz.open(metadata.get('__path'))
    pages = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
    doc.close()
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    for page_text in pages:
        all_chunks.extend(splitter.split_text(page_text))
    return all_chunks

def semantic_chunk(text: str, metadata: dict) -> List[str]:
    """
    Usa embeddings de sentença + KMeans para agrupar em CHUNK_COUNT clusters.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = text.split('. ')
    embeddings = model.encode(sentences)
    n_clusters = max(1, len(sentences) // 10)
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(embeddings)
    clusters = {i: [] for i in range(n_clusters)}
    for sent, label in zip(sentences, labels):
        clusters[label].append(sent)
    return ['. '.join(cluster) for cluster in clusters.values()]

def recursive_chunk(text: str, metadata: dict) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)
    
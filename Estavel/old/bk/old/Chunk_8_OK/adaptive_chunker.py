import logging, re
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, SLIDING_WINDOW_OVERLAP_RATIO, SIMILARITY_THRESHOLD
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from sentence_transformers import SentenceTransformer, util

# Carrega modelo SBERT para semantic chunking (instância única)
_sbert = SentenceTransformer('all-MiniLM-L6-v2')


def filter_paragraphs(text: str) -> List[str]:
    """
    Descarta sumários e parágrafos muito curtos (<50 caracteres).
    """
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    result = []
    for p in paras:
        low = p.lower()
        if re.search(r'\b(sum[aá]rio|índice|contents?|table of contents)\b', low):
            continue
        if len(p) < 50:
            continue
        result.append(p)
    return result


def sliding_window_chunk(p: str, window_size: int, overlap: int) -> List[str]:
    """
    Gera chunks com sliding window de tokens.
    """
    tokens = p.split()
    stride = max(1, window_size - overlap)
    chunks = []
    for i in range(0, len(tokens), stride):
        part = tokens[i:i + window_size]
        if not part:
            break
        chunks.append(' '.join(part))
        if i + window_size >= len(tokens):
            break
    return chunks


def semantic_chunk(paras: List[str]) -> List[str]:
    """
    Agrupa sentenças semanticamente similares usando SBERT.
    """
    # computa embeddings de cada parágrafo
    embs = _sbert.encode(paras, convert_to_tensor=True)
    clusters = []
    used = set()
    for i, emb in enumerate(embs):
        if i in used:
            continue
        group = [paras[i]]
        used.add(i)
        # encontra similares acima do threshold
        sims = util.cos_sim(emb, embs).tolist()[0]
        for j, score in enumerate(sims):
            if j != i and score >= SIMILARITY_THRESHOLD:
                group.append(paras[j])
                used.add(j)
        clusters.append(' '.join(group))
    return clusters


def hierarchical_chunk(text: str, metadata: dict) -> List[str]:
    """
    Pipeline multi-nível:
    1) Filtra parágrafos relevantes
    2) Agrupa semanticamente
    3) Para cada cluster/parágrafo:
       - se <= CHUNK_SIZE: mantém
       - se > CHUNK_SIZE: sliding window com overlap adaptativo
    """
    paras = filter_paragraphs(text)
    # Passo 1: semantic chunking para agrupar contextos próximos
    sem_chunks = semantic_chunk(paras)
    final_chunks: List[str] = []
    for chunk in sem_chunks:
        if len(chunk) <= CHUNK_SIZE:
            final_chunks.append(chunk)
        else:
            # overlap máximo 50% do texto ou configurado
            max_ov = min(CHUNK_OVERLAP, int(len(chunk) * SLIDING_WINDOW_OVERLAP_RATIO))
            parts = sliding_window_chunk(chunk, CHUNK_SIZE, max_ov)
            if not parts:
                logging.warning(f"Falha sliding_window para chunk len={len(chunk)}")
                parts = TokenTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=max_ov
                ).split_text(chunk)
            final_chunks.extend(parts)
    return final_chunks
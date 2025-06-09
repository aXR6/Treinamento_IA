import logging
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter


def filter_paragraphs(text: str) -> list[str]:
    """
    Retorna apenas parágrafos relevantes, descartando sumário e trechos muito curtos.
    """
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    result = []
    for p in paras:
        low = p.lower()
        # descarta sumário, índice e table of contents
        if re.search(r'\b(sum[aá]rio|índice|contents?|table of contents)\b', low):
            continue
        # descarta parágrafos muito curtos (<50 chars)
        if len(p) < 50:
            continue
        result.append(p)
    return result


def adaptive_chunk(text: str, metadata: dict) -> list[str]:
    """
    Para cada parágrafo filtrado:
    - se <= CHUNK_SIZE: mantém inteiro
    - se > CHUNK_SIZE: divide respeitando overlap máximo de 50% do parágrafo
    """
    paras = filter_paragraphs(text)
    chunks: list[str] = []
    for p in paras:
        if len(p) <= CHUNK_SIZE:
            chunks.append(p)
        else:
            overlap = min(CHUNK_OVERLAP, len(p) // 2)
            splitter = RecursiveCharacterTextSplitter(
                separators=SEPARATORS,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=overlap
            )
            sub = splitter.split_text(p)
            if not sub:
                logging.warning(f"Recursive split vazio para parágrafo longo (len={len(p)}); fallback token-aware")
                token_splitter = TokenTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=overlap
                )
                sub = token_splitter.split_text(p)
            chunks.extend(sub)
    return chunks
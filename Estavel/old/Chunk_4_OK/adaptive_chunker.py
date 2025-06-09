# adaptive_chunker.py
"""
Chunking semântico adaptativo para PDFs e documentos diversos.
Combina:
 - RecursiveCharacterTextSplitter para respeitar limites de parágrafo/título
 - TokenTextSplitter como fallback token-aware
"""
import logging

from config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)


def adaptive_chunk(text: str, metadata: dict) -> list[str]:
    """
    Divide o texto em pedaços coerentes para gerar embeddings.
    1) Tenta uma divisão recursiva respeitando parágrafos e sentenças.
    2) Se falhar, recorre a divisão baseada em tokens.
    Args:
        text: Texto completo a ser chunked.
        metadata: Dicionário de metadados (pode influenciar estratégia, futuro).
    Returns:
        Lista de strings, cada uma um chunk adequado.
    """
    # 1) Splitter recursivo (prioriza parágrafos e sentenças)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ","],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    try:
        chunks = splitter.split_text(text)
        if not chunks:
            raise ValueError("Recursive splitter retornou lista vazia")
        return chunks
    except Exception as e:
        logging.warning(
            f"RecursiveCharacterTextSplitter falhou ({e}), tentando TokenTextSplitter..."
        )
        # 2) Fallback token-aware
        token_splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return token_splitter.split_text(text)

# paragraph_group_splitter.py
from typing import List

class ParagraphGroupSplitter:
    """
    Agrupa N parágrafos em um único chunk, com overlap de M parágrafos.
    """
    def __init__(self, group_size: int, overlap: int):
        self.group_size = group_size
        self.overlap = overlap

    def split(self, paragraphs: List[str]) -> List[str]:
        chunks: List[str] = []
        step = max(1, self.group_size - self.overlap)
        for i in range(0, len(paragraphs), step):
            group = paragraphs[i : i + self.group_size]
            if group:
                chunks.append("\n\n".join(group))
        return chunks
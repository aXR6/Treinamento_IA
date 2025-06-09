# new_paragraph_splitter.py
from langchain.text_splitter import CharacterTextSplitter

class ParagraphSplitter:
    """
    Separa o texto exatamente por parágrafos (dupla quebra de linha).
    """
    def __init__(self, chunk_size: int, overlap: int):
        self.splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def split(self, text: str) -> list[str]:
        paras = self.splitter.split_text(text)
        return [p.strip() for p in paras if p.strip()]
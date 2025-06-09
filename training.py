# training.py
import logging
import psycopg2
from typing import List

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE


def _fetch_texts(dim: int) -> List[str]:
    """Lê a coluna `content` de public.documents_<dim>."""
    table = f"public.documents_{dim}"
    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        cur = conn.cursor()
        cur.execute(f"SELECT content FROM {table}")
        rows = cur.fetchall()
        cur.close()
        return [r[0] for r in rows]
    except Exception as e:
        logging.error(f"Erro ao ler dados de {table}: {e}")
        return []
    finally:
        if conn:
            conn.close()


def train_model(model_name: str, dim: int, device: str) -> None:
    """Ajusta modelo SBERT usando textos do PostgreSQL."""
    texts = _fetch_texts(dim)
    if not texts:
        logging.error("Nenhum texto encontrado para treinamento.")
        return

    logging.info(f"Carregando modelo '{model_name}' em {device}…")
    model = SentenceTransformer(model_name, device=device)

    train_examples = [InputExample(texts=[t, t]) for t in texts]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    logging.info(f"Iniciando treinamento com {len(train_examples)} exemplos…")
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, show_progress_bar=True)

    out_path = f"{model_name.replace('/', '_')}_finetuned_{dim}"
    model.save(out_path)
    logging.info(f"Modelo salvo em {out_path}")


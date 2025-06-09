# training.py
import logging
from typing import List, Optional

import psycopg2
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from config import (
    PG_HOST,
    PG_PORT,
    PG_USER,
    PG_PASSWORD,
    PG_DATABASE,
    TRAINING_MODEL_NAME,
)


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


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def train_model(dim: int, device: str, model_name: Optional[str] = None) -> None:
    """Ajusta um modelo Hugging Face usando textos do PostgreSQL."""
    texts = _fetch_texts(dim)
    if not texts:
        logging.error("Nenhum texto encontrado para treinamento.")
        return

    base_model = model_name or TRAINING_MODEL_NAME
    logging.info(f"Carregando modelo '{base_model}'…")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=tokenizer.model_max_length
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"{base_model.replace('/', '_')}_finetuned_{dim}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=10,
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model.to(_resolve_device(device)),
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    logging.info("Iniciando treinamento…")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    logging.info(f"Modelo salvo em {training_args.output_dir}")


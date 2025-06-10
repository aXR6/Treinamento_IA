# training.py
#!/usr/bin/env python3
import logging
from typing import Iterable, Iterator, Optional
import os

import psycopg2
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from tqdm import tqdm
from datasets import Dataset, Features, Value

from config import (
    PG_HOST,
    PG_PORT,
    PG_USER,
    PG_PASSWORD,
    PG_DATABASE,
    TRAINING_MODEL_NAME,
)

def _fetch_texts(dim: int, batch_size: int = 1000) -> Iterator[str]:
    """Lê a coluna `content` de public.documents_<dim>` de forma preguiçosa."""
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
        # Usa cursor no servidor para evitar carregar todos os resultados em memória
        with conn.cursor(name=f"cursor_{dim}") as cur:
            cur.itersize = batch_size
            cur.execute(f"SELECT content FROM {table}")
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield row[0]
    except Exception as e:
        logging.error(f"Erro ao ler dados de {table}: {e}")
    finally:
        if conn:
            conn.close()

def _should_use_cuda(device_str: str, allow_gpu: bool) -> bool:
    """Decide se o treinamento deve usar CUDA."""
    if device_str == "gpu":
        return torch.cuda.is_available()
    if device_str == "auto" and allow_gpu and torch.cuda.is_available():
        return True
    return False

def _resolve_device(device_str: str, allow_gpu: bool) -> str:
    return "cuda" if _should_use_cuda(device_str, allow_gpu) else "cpu"


class ProgressCallback(TrainerCallback):
    """Exibe barra de progresso e métricas durante o treinamento."""

    def __init__(self) -> None:
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        total = state.max_steps
        self.pbar = tqdm(total=total, desc="Treinando", unit="step")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pbar and logs:
            info = {}
            if "loss" in logs:
                info["loss"] = f"{logs['loss']:.4f}"
            if "epoch" in logs:
                info["epoch"] = f"{logs['epoch']:.2f}"
            if info:
                self.pbar.set_postfix(info)

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()
            self.pbar = None

def train_model(
    dim: int,
    device: str,
    model_name: Optional[str] = None,
    allow_auto_gpu: bool = True,
    epochs: int = 1,
    batch_size: int = 1,
) -> None:
    """Ajusta um modelo Hugging Face usando textos do PostgreSQL."""
    texts_iter = _fetch_texts(dim)
    try:
        first_text = next(texts_iter)
    except StopIteration:
        logging.error("Nenhum texto encontrado para treinamento.")
        return

    def text_generator() -> Iterator[dict]:
        yield {"text": first_text}
        for txt in texts_iter:
            yield {"text": txt}

    use_cuda = _should_use_cuda(device, allow_auto_gpu)
    prev_env = os.environ.get("TRANSFORMERS_NO_CUDA")
    if not use_cuda:
        os.environ["TRANSFORMERS_NO_CUDA"] = "1"
    else:
        if prev_env is not None:
            os.environ.pop("TRANSFORMERS_NO_CUDA", None)

    base_model = model_name or TRAINING_MODEL_NAME
    logging.info(f"Carregando modelo '{base_model}'…")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    features = Features({"text": Value("string")})
    dataset = Dataset.from_generator(text_generator, features=features)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=tokenizer.model_max_length
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    resolved_device = _resolve_device(device, allow_auto_gpu)

    training_args = TrainingArguments(
        output_dir=f"{base_model.replace('/', '_')}_finetuned_{dim}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=10,
        overwrite_output_dir=True,
        use_cpu=resolved_device == "cpu",
    )

    trainer = Trainer(
        model=model.to(resolved_device),
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[ProgressCallback()],
    )

    logging.info("Iniciando treinamento…")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    logging.info(f"Modelo salvo em {training_args.output_dir}")

    # Restaura variável de ambiente original
    if prev_env is not None:
        os.environ["TRANSFORMERS_NO_CUDA"] = prev_env
    else:
        os.environ.pop("TRANSFORMERS_NO_CUDA", None)

# training.py
#!/usr/bin/env python3
import logging
from typing import Iterable, Iterator, Optional
import os
import inspect

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
    PG_DB_PDF,
    PG_DB_QA,
    TRAINING_MODEL_NAME,
)

def _fetch_texts(dim: int, db_name: str, batch_size: int = 1000) -> Iterator[str]:
    """Lê a coluna `content` de public.documents_<dim>` de forma preguiçosa."""
    table = f"public.documents_{dim}"
    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=db_name,
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

def _fetch_qa_pairs(dim: int, db_name: str, batch_size: int = 1000) -> Iterator[tuple[str, str, str]]:
    """Lê colunas `content`, `question` e `answer` de public.documents_<dim>."""
    table = f"public.documents_{dim}"
    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=db_name,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        with conn.cursor(name=f"cursor_qa_{dim}") as cur:
            cur.itersize = batch_size
            cur.execute(
                f"SELECT content, question, answer FROM {table} "
                "WHERE question IS NOT NULL AND answer IS NOT NULL "
                "AND question <> '' AND answer <> ''"
            )
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for content, q, a in rows:
                    yield content, q, a
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
            # Imprime dicionário completo de logs para maior interação
            self.pbar.write(str(logs))

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
    eval_steps: int = 500,
    validation_split: float = 0.1,
    max_seq_length: int = 0,
) -> None:
    """Ajusta um modelo Hugging Face usando textos do PostgreSQL.

    Parameters
    ----------
    eval_steps : int
        Avalia o modelo a cada ``eval_steps`` passos.
    validation_split : float
        Porcentagem do dataset reservada para validação.
    max_seq_length : int
        Comprimento máximo das sequências (0 usa ``tokenizer.model_max_length``).
    """
    def text_generator() -> Iterator[dict]:
        for txt in _fetch_texts(dim, PG_DB_PDF):
            yield {"text": txt}

    use_cuda = _should_use_cuda(device, allow_auto_gpu)
    prev_env = os.environ.get("TRANSFORMERS_NO_CUDA")
    if not use_cuda:
        os.environ["TRANSFORMERS_NO_CUDA"] = "1"
    else:
        if prev_env is not None:
            os.environ.pop("TRANSFORMERS_NO_CUDA", None)

    try:
        base_model = model_name or TRAINING_MODEL_NAME
        logging.info(f"Carregando modelo '{base_model}'…")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)

        features = Features({"text": Value("string")})
        try:
            dataset = Dataset.from_generator(text_generator, features=features)
        except Exception:
            logging.error("Nenhum texto encontrado para treinamento.")
            return
        dataset_len = len(dataset)
        logging.info(f"Total de textos carregados: {dataset_len}")

        def tokenize_fn(examples):
            max_len = max_seq_length or tokenizer.model_max_length
            return tokenizer(examples["text"], truncation=True, max_length=max_len)

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        split = tokenized.train_test_split(test_size=validation_split, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        resolved_device = _resolve_device(device, allow_auto_gpu)
        logging.info(f"Dispositivo escolhido: {resolved_device}")

        out_dir = f"{base_model.replace('/', '_')}_finetuned_{dim}"
        base_args = {
            "output_dir": out_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "overwrite_output_dir": True,
            "use_cpu": resolved_device == "cpu",
        }

        sig = inspect.signature(TrainingArguments)
        if "evaluation_strategy" in sig.parameters:
            base_args.update({
                "per_device_eval_batch_size": batch_size,
                "logging_steps": 10,
                "evaluation_strategy": "steps",
                "eval_steps": eval_steps,
                "save_strategy": "steps",
                "save_steps": eval_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "loss",
                "greater_is_better": False,
            })
        else:
            base_args["logging_steps"] = 10

        training_args = TrainingArguments(**base_args)

        try:
            model = model.to(resolved_device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"Memória insuficiente para mover modelo: {e}")
                print(
                    "\n⚠️  Não há memória de vídeo suficiente. "
                    "Reduza o batch size ou selecione 'cpu' como dispositivo."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            raise

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=[ProgressCallback()],
        )

        logging.info("Iniciando treinamento…")
        try:
            trainer.train()
            trainer.save_model(training_args.output_dir)
            best_dir = os.path.join(training_args.output_dir, "best_model")
            trainer.save_model(best_dir)
            logging.info(
                f"Modelo salvo em {training_args.output_dir} (melhor em {best_dir})"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"Memória insuficiente durante o treinamento: {e}")
                print(
                    "\n⚠️  A GPU ficou sem memória durante o treinamento. "
                    "Tente reduzir o batch size ou utilize a CPU."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            raise
    finally:
        if prev_env is not None:
            os.environ["TRANSFORMERS_NO_CUDA"] = prev_env
        else:
            os.environ.pop("TRANSFORMERS_NO_CUDA", None)


def train_qa_model(
    dim: int,
    device: str,
    model_name: Optional[str] = None,
    allow_auto_gpu: bool = True,
    epochs: int = 1,
    batch_size: int = 1,
    eval_steps: int = 500,
    validation_split: float = 0.1,
    max_seq_length: int = 0,
) -> None:
    """Ajusta modelo usando pares pergunta/resposta do PostgreSQL."""

    def text_generator() -> Iterator[dict]:
        for content, q, a in _fetch_qa_pairs(dim, PG_DB_QA):
            yield {"text": f"Pergunta: {q}\nContexto: {content}\nResposta: {a}"}

    use_cuda = _should_use_cuda(device, allow_auto_gpu)
    prev_env = os.environ.get("TRANSFORMERS_NO_CUDA")
    if not use_cuda:
        os.environ["TRANSFORMERS_NO_CUDA"] = "1"
    else:
        if prev_env is not None:
            os.environ.pop("TRANSFORMERS_NO_CUDA", None)

    try:
        base_model = model_name or TRAINING_MODEL_NAME
        logging.info(f"Carregando modelo '{base_model}'…")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)

        features = Features({"text": Value("string")})
        try:
            dataset = Dataset.from_generator(text_generator, features=features)
        except Exception:
            logging.error("Nenhum par QA encontrado para treinamento.")
            return

        dataset_len = len(dataset)
        logging.info(f"Total de pares carregados: {dataset_len}")

        def tokenize_fn(examples):
            max_len = max_seq_length or tokenizer.model_max_length
            return tokenizer(examples["text"], truncation=True, max_length=max_len)

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        split = tokenized.train_test_split(test_size=validation_split, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        resolved_device = _resolve_device(device, allow_auto_gpu)
        logging.info(f"Dispositivo escolhido: {resolved_device}")

        out_dir = f"{base_model.replace('/', '_')}_finetuned_qa_{dim}"
        base_args = {
            "output_dir": out_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "overwrite_output_dir": True,
            "use_cpu": resolved_device == "cpu",
        }

        sig = inspect.signature(TrainingArguments)
        if "evaluation_strategy" in sig.parameters:
            base_args.update({
                "per_device_eval_batch_size": batch_size,
                "logging_steps": 10,
                "evaluation_strategy": "steps",
                "eval_steps": eval_steps,
                "save_strategy": "steps",
                "save_steps": eval_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "loss",
                "greater_is_better": False,
            })
        else:
            base_args["logging_steps"] = 10

        training_args = TrainingArguments(**base_args)

        try:
            model = model.to(resolved_device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"Memória insuficiente para mover modelo: {e}")
                print(
                    "\n⚠️  Não há memória de vídeo suficiente. "
                    "Reduza o batch size ou selecione 'cpu' como dispositivo."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            raise

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=[ProgressCallback()],
        )

        logging.info("Iniciando treinamento…")
        try:
            trainer.train()
            trainer.save_model(training_args.output_dir)
            best_dir = os.path.join(training_args.output_dir, "best_model")
            trainer.save_model(best_dir)
            logging.info(
                f"Modelo salvo em {training_args.output_dir} (melhor em {best_dir})"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"Memória insuficiente durante o treinamento: {e}")
                print(
                    "\n⚠️  A GPU ficou sem memória durante o treinamento. "
                    "Tente reduzir o batch size ou utilize a CPU."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            raise
    finally:
        if prev_env is not None:
            os.environ["TRANSFORMERS_NO_CUDA"] = prev_env
        else:
            os.environ.pop("TRANSFORMERS_NO_CUDA", None)

#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Optional
import time
import logging
from tqdm import tqdm
import torch

# Garante imports locais
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OLLAMA_EMBEDDING_MODEL, SERAFIM_EMBEDDING_MODEL,
    MINILM_L6_V2, MINILM_L12_V2, MPNET_EMBEDDING_MODEL,
    DIM_MXBAI, DIM_SERAFIM, DIM_MINILM_L6, DIM_MINIL12, DIM_MPNET,
    OCR_THRESHOLD, EVAL_STEPS, VALIDATION_SPLIT, MAX_SEQ_LENGTH,
    PG_DB_PDF, PG_DB_QA, PG_DB_CVE,
    TOKENIZE_NUM_PROC, DATALOADER_NUM_WORKERS,
    validate_config
)
from extractors import extract_text
from utils import setup_logging, is_valid_file, build_record, move_to_processed
from constants import VALID_EXTS
from pg_storage import save_to_postgres
from metrics import start_metrics_server

# Valida configuração e inicializa logs e métricas
validate_config()
setup_logging()
start_metrics_server()

# Opções de menu
STRATEGY_OPTIONS = [
    "pypdf", "pdfminer", "pdfminer-low", "unstructured",
    "ocr", "plumber", "pymupdf4llm"
]
EMBED_MODELS = {
    "1": OLLAMA_EMBEDDING_MODEL,
    "2": SERAFIM_EMBEDDING_MODEL,
    "3": MINILM_L6_V2,
    "4": MINILM_L12_V2,
    "5": MPNET_EMBEDDING_MODEL
}
DIMENSIONS = {
    "1": DIM_MXBAI,
    "2": DIM_SERAFIM,
    "3": DIM_MINILM_L6,
    "4": DIM_MINIL12,
    "5": DIM_MPNET
}
DB_OPTIONS = {
    "1": PG_DB_PDF,
    "2": PG_DB_QA,
    "3": PG_DB_CVE,
}

def _resolve_dev(device: str) -> str:
    if device == "gpu" and torch.cuda.is_available():
        return "cuda"
    if device in ("gpu", "auto") and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def test_model(path: str, device: str) -> None:
    """Carrega modelo e permite gerar texto interativamente."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        logging.error(f"Falha ao importar transformers: {e}")
        print("Dependência 'transformers' ausente")
        return

    dev = _resolve_dev(device)
    try:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(dev)
    except Exception as e:
        logging.error(f"Falha ao carregar modelo: {e}")
        print("Não foi possível carregar o modelo informado")
        return

    while True:
        prompt = input("Prompt (ENTER para sair): ").strip()
        if not prompt:
            break
        data = tok(prompt, return_tensors="pt")
        data = {k: v.to(model.device) for k, v in data.items()}
        with torch.no_grad():
            out_ids = model.generate(**data, max_new_tokens=128)
        print(tok.decode(out_ids[0], skip_special_tokens=True))


def chat_model(path: str, device: str) -> None:
    """Inicia uma conversa mantendo o hist\u00f3rico de mensagens."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        logging.error(f"Falha ao importar transformers: {e}")
        print("Depend\u00eancia 'transformers' ausente")
        return

    dev = _resolve_dev(device)
    try:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(dev)
    except Exception as e:
        logging.error(f"Falha ao carregar modelo: {e}")
        print("N\u00e3o foi poss\u00edvel carregar o modelo informado")
        return

    history = ""
    while True:
        prompt = input("Voc\u00ea: ").strip()
        if not prompt:
            break
        history += f"Usu\u00e1rio: {prompt}\nAssistente:"
        data = tok(history, return_tensors="pt")
        data = {k: v.to(model.device) for k, v in data.items()}
        with torch.no_grad():
            out_ids = model.generate(**data, max_new_tokens=128)
        if hasattr(data["input_ids"], "shape"):
            in_len = data["input_ids"].shape[1]
        elif isinstance(data["input_ids"], (list, tuple)):
            elem = data["input_ids"][0]
            in_len = len(elem) if isinstance(elem, (list, tuple)) else len(data["input_ids"])
        else:
            in_len = 0
        new_tokens = out_ids[0][in_len:]
        response = tok.decode(new_tokens, skip_special_tokens=True)
        history += f" {response}\n"
        print(response)

def model_test_menu(current_path: str, device: str) -> tuple[str, str]:
    while True:
        clear_screen()
        print("*** Menu Teste de Modelo ***")
        show = current_path or "(indefinido)"
        print(f"1 - Caminho do modelo (atual: {show})")
        print(f"2 - Dispositivo (atual: {device})")
        print("3 - Iniciar teste")
        print("4 - Conversar com modelo")
        print("0 - Voltar")
        c = input("> ").strip()

        if c == "10":
            break
        elif c == "1":
            p = input(f"Diretório do modelo [{current_path}]: ").strip()
            if p:
                current_path = p
        elif c == "2":
            device = select_device(device)
        elif c == "3":
            if not current_path:
                print("Caminho não definido.")
                input("ENTER para continuar…")
                continue
            test_model(current_path, device)
            input("ENTER para continuar…")
        elif c == "4":
            if not current_path:
                print("Caminho não definido.")
                input("ENTER para continuar…")
                continue
            chat_model(current_path, device)
            input("ENTER para continuar…")
        else:
            print("Opção inválida.")
            time.sleep(1)

    return current_path, device

def select_device(current: str) -> str:
    print("\n*** Selecione Dispositivo ***")
    options = ["cpu", "auto"]
    if torch.cuda.is_available():
        options.insert(1, "gpu")
    for i, opt in enumerate(options, 1):
        print(f"{i} - {opt}")
    c = input(f"Escolha [{current}]: ").strip()
    if c.isdigit() and 1 <= int(c) <= len(options):
        return options[int(c)-1]
    return current

def toggle_tf_cuda(current: bool) -> bool:
    print("\n*** Transformers deve detectar GPU automaticamente? ***")
    print("1 - Sim")
    print("2 - Não")
    c = input(f"Escolha [{'Sim' if current else 'Não'}]: ").strip()
    if c == "1":
        return True
    if c == "2":
        return False
    return current

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def select_strategy(current: str) -> str:
    print("\n*** Selecione Estratégia ***")
    for i, k in enumerate(STRATEGY_OPTIONS, 1):
        print(f"{i} - {k}")
    c = input(f"Escolha [{current}]: ").strip()
    if c.isdigit() and 1 <= int(c) <= len(STRATEGY_OPTIONS):
        return STRATEGY_OPTIONS[int(c)-1]
    return current

def select_embedding(current: str) -> str:
    print("\n*** Selecione Embedding ***")
    for k, n in EMBED_MODELS.items():
        print(f"{k} - {n}")
    c = input(f"Escolha [{current}]: ").strip()
    return EMBED_MODELS.get(c, current)

def select_dimension(current: int) -> int:
    print("\n*** Selecione Dimensão ***")
    for k, d in DIMENSIONS.items():
        print(f"{k} - {d}")
    c = input(f"Escolha [{current}]: ").strip()
    return DIMENSIONS.get(c, current)

def select_database(current: str) -> str:
    print("\n*** Selecione Banco de Dados ***")
    for k, d in DB_OPTIONS.items():
        print(f"{k} - {d}")
    c = input(f"Escolha [{current}]: ").strip()
    return DB_OPTIONS.get(c, current)

def training_menu(
    train_dim: int,
    device: str,
    allow_tf_cuda: bool,
    epochs: int,
    batch_size: int,
    eval_steps: int,
    val_split: float,
    tokenize_num_proc: int,
    dataloader_num_workers: int,
) -> tuple[int, bool, int, int, int, float, int, int]:
    """Exibe o submenu de treinamento."""
    while True:
        clear_screen()
        print("*** Menu Treinamento ***")
        print("1 - Treinar modelo")
        print(
            f"2 - Transformers detectar GPU automaticamente: {'Sim' if allow_tf_cuda else 'Não'}"
        )
        print(f"3 - Selecionar tabela (atual: documents_{train_dim})")
        print(f"4 - Épocas (atual: {epochs})")
        print(f"5 - Batch size (atual: {batch_size})")
        print(f"6 - Avaliar a cada N passos (atual: {eval_steps})")
        print(f"7 - Porcentagem validação (atual: {val_split})")
        print(f"8 - Processos de tokenização (atual: {tokenize_num_proc})")
        print(f"9 - Workers DataLoader (atual: {dataloader_num_workers})")
        print("0 - Voltar")
        c = input("> ").strip()

        if c == "0":
            break

        elif c == "1":
            from training import train_model
            train_model(
                train_dim,
                device,
                allow_auto_gpu=allow_tf_cuda,
                epochs=epochs,
                batch_size=batch_size,
                eval_steps=eval_steps,
                validation_split=val_split,
                max_seq_length=MAX_SEQ_LENGTH,
                tokenize_num_proc=tokenize_num_proc,
                dataloader_num_workers=dataloader_num_workers,
            )
            input("ENTER para continuar…")

        elif c == "2":
            allow_tf_cuda = toggle_tf_cuda(allow_tf_cuda)

        elif c == "3":
            train_dim = select_dimension(train_dim)

        elif c == "4":
            inp = input(f"Número de épocas [{epochs}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                epochs = int(inp)

        elif c == "5":
            inp = input(f"Batch size [{batch_size}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                batch_size = int(inp)

        elif c == "6":
            inp = input(f"Avaliar a cada quantos passos? [{eval_steps}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                eval_steps = int(inp)

        elif c == "7":
            inp = input(f"Porcentagem de valida\u00e7\u00e3o (0-1) [{val_split}]: ").strip()
            try:
                v = float(inp)
                if 0 < v < 1:
                    val_split = v
            except ValueError:
                pass

        else:
            print("Opção inválida.")
            time.sleep(1)

    return (
        train_dim,
        allow_tf_cuda,
        epochs,
        batch_size,
        eval_steps,
        val_split,
        tokenize_num_proc,
        dataloader_num_workers,
    )


def cve_training_menu(
    device: str,
    allow_tf_cuda: bool,
    epochs: int,
    batch_size: int,
    eval_steps: int,
    val_split: float,
    tokenize_num_proc: int,
    dataloader_num_workers: int,
) -> tuple[bool, int, int, int, float, int, int]:
    """Menu de treinamento usando base CVE."""
    while True:
        clear_screen()
        print("*** Menu Treinamento CVE ***")
        print("1 - Treinar modelo")
        print(
            f"2 - Transformers detectar GPU automaticamente: {'Sim' if allow_tf_cuda else 'Não'}"
        )
        print(f"3 - Épocas (atual: {epochs})")
        print(f"4 - Batch size (atual: {batch_size})")
        print(f"5 - Avaliar a cada N passos (atual: {eval_steps})")
        print(f"6 - Porcentagem validação (atual: {val_split})")
        print(f"7 - Processos de tokenização (atual: {tokenize_num_proc})")
        print(f"8 - Workers DataLoader (atual: {dataloader_num_workers})")
        print("0 - Voltar")
        c = input("> ").strip()

        if c == "0":
            break
        elif c == "1":
            from training import train_cve_model
            train_cve_model(
                device,
                allow_auto_gpu=allow_tf_cuda,
                epochs=epochs,
                batch_size=batch_size,
                eval_steps=eval_steps,
                validation_split=val_split,
                max_seq_length=MAX_SEQ_LENGTH,
                tokenize_num_proc=tokenize_num_proc,
                dataloader_num_workers=dataloader_num_workers,
            )
            input("ENTER para continuar…")
        elif c == "2":
            allow_tf_cuda = toggle_tf_cuda(allow_tf_cuda)
        elif c == "3":
            inp = input(f"Número de épocas [{epochs}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                epochs = int(inp)
        elif c == "4":
            inp = input(f"Batch size [{batch_size}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                batch_size = int(inp)
        elif c == "5":
            inp = input(f"Avaliar a cada quantos passos? [{eval_steps}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                eval_steps = int(inp)
        elif c == "6":
            inp = input(f"Porcentagem de validação (0-1) [{val_split}]: ").strip()
            try:
                v = float(inp)
                if 0 < v < 1:
                    val_split = v
            except ValueError:
                pass
        elif c == "7":
            inp = input(f"Tokenize num_proc [{tokenize_num_proc}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                tokenize_num_proc = int(inp)
        elif c == "8":
            inp = input(f"Dataloader workers [{dataloader_num_workers}]: ").strip()
            if inp.isdigit() and int(inp) >= 0:
                dataloader_num_workers = int(inp)
        else:
            print("Opção inválida.")
            time.sleep(1)

    return (
        allow_tf_cuda,
        epochs,
        batch_size,
        eval_steps,
        val_split,
        tokenize_num_proc,
        dataloader_num_workers,
    )


def qa_training_menu(
    train_dim: int,
    device: str,
    allow_tf_cuda: bool,
    epochs: int,
    batch_size: int,
    eval_steps: int,
    val_split: float,
    tokenize_num_proc: int,
    dataloader_num_workers: int,
) -> tuple[int, bool, int, int, int, float, int, int]:
    """Menu de treinamento para perguntas e respostas."""
    while True:
        clear_screen()
        print("*** Menu Treinamento QA ***")
        print("1 - Treinar modelo")
        print(
            f"2 - Transformers detectar GPU automaticamente: {'Sim' if allow_tf_cuda else 'Não'}"
        )
        print(f"3 - Selecionar tabela (atual: documents_{train_dim})")
        print(f"4 - Épocas (atual: {epochs})")
        print(f"5 - Batch size (atual: {batch_size})")
        print(f"6 - Avaliar a cada N passos (atual: {eval_steps})")
        print(f"7 - Porcentagem validação (atual: {val_split})")
        print(f"8 - Processos de tokenização (atual: {tokenize_num_proc})")
        print(f"9 - Workers DataLoader (atual: {dataloader_num_workers})")
        print("0 - Voltar")
        c = input("> ").strip()

        if c == "0":
            break
        elif c == "1":
            from training import train_qa_model
            train_qa_model(
                train_dim,
                device,
                allow_auto_gpu=allow_tf_cuda,
                epochs=epochs,
                batch_size=batch_size,
                eval_steps=eval_steps,
                validation_split=val_split,
                max_seq_length=MAX_SEQ_LENGTH,
                tokenize_num_proc=tokenize_num_proc,
                dataloader_num_workers=dataloader_num_workers,
            )
            input("ENTER para continuar…")
        elif c == "2":
            allow_tf_cuda = toggle_tf_cuda(allow_tf_cuda)
        elif c == "3":
            train_dim = select_dimension(train_dim)
        elif c == "4":
            inp = input(f"Número de épocas [{epochs}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                epochs = int(inp)
        elif c == "5":
            inp = input(f"Batch size [{batch_size}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                batch_size = int(inp)
        elif c == "6":
            inp = input(f"Avaliar a cada quantos passos? [{eval_steps}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                eval_steps = int(inp)
        elif c == "7":
            inp = input(f"Porcentagem de validação (0-1) [{val_split}]: ").strip()
            try:
                v = float(inp)
                if 0 < v < 1:
                    val_split = v
            except ValueError:
                pass

        elif c == "8":
            inp = input(f"Tokenize num_proc [{tokenize_num_proc}]: ").strip()
            if inp.isdigit() and int(inp) > 0:
                tokenize_num_proc = int(inp)

        elif c == "9":
            inp = input(f"Dataloader workers [{dataloader_num_workers}]: ").strip()
            if inp.isdigit() and int(inp) >= 0:
                dataloader_num_workers = int(inp)
        else:
            print("Opção inválida.")
            time.sleep(1)

    return (
        train_dim,
        allow_tf_cuda,
        epochs,
        batch_size,
        eval_steps,
        val_split,
        tokenize_num_proc,
        dataloader_num_workers,
    )


def training_type_menu(
    train_dim: int,
    device: str,
    allow_tf_cuda: bool,
    epochs: int,
    batch_size: int,
    eval_steps: int,
    val_split: float,
    tokenize_num_proc: int,
    dataloader_num_workers: int,
) -> tuple[int, bool, int, int, int, float, int, int]:
    """Menu para escolher o tipo de treinamento."""
    while True:
        clear_screen()
        print("*** Tipo de Treinamento ***")
        print("1 - Treinamento")
        print("2 - Treinamento QA")
        print("3 - Treinamento CVE")
        print("0 - Voltar")
        c = input("> ").strip()

        if c == "0":
            break
        elif c == "1":
            (
                train_dim,
                allow_tf_cuda,
                epochs,
                batch_size,
                eval_steps,
                val_split,
                tokenize_num_proc,
                dataloader_num_workers,
            ) = training_menu(
                train_dim,
                device,
                allow_tf_cuda,
                epochs,
                batch_size,
                eval_steps,
                val_split,
                tokenize_num_proc,
                dataloader_num_workers,
            )
        elif c == "2":
            (
                train_dim,
                allow_tf_cuda,
                epochs,
                batch_size,
                eval_steps,
                val_split,
                tokenize_num_proc,
                dataloader_num_workers,
            ) = qa_training_menu(
                train_dim,
                device,
                allow_tf_cuda,
                epochs,
                batch_size,
                eval_steps,
                val_split,
                tokenize_num_proc,
                dataloader_num_workers,
            )
        elif c == "3":
            (
                allow_tf_cuda,
                epochs,
                batch_size,
                eval_steps,
                val_split,
                tokenize_num_proc,
                dataloader_num_workers,
            ) = cve_training_menu(
                device,
                allow_tf_cuda,
                epochs,
                batch_size,
                eval_steps,
                val_split,
                tokenize_num_proc,
                dataloader_num_workers,
            )
        else:
            print("Opção inválida.")
            time.sleep(1)

    return (
        train_dim,
        allow_tf_cuda,
        epochs,
        batch_size,
        eval_steps,
        val_split,
        tokenize_num_proc,
        dataloader_num_workers,
    )

def process_file(path: str, strat: str, model: str, dim: int, device: str,
                 db_name: str, stats: dict, processed_root: Optional[str] = None):
    """
    Processa um único arquivo: extrai texto, gera embeddings e salva no PostgreSQL.
    Agora o save_to_postgres retorna a lista completa de registros inseridos,
    para que possamos logar quantos chunks foram inseridos e (se aplicável)
    qual a pontuação de reranking de cada um.
    """
    filename = os.path.basename(path)
    logging.info(f"→ Processando arquivo: {filename}  |  Estratégia: {strat}  |  Embedding: {model}  |  Dimensão: {dim}")

    p = os.path.normpath(path.strip())
    base, ext = os.path.splitext(p)
    p2 = base.rstrip() + ext
    if p2 != p:
        try:
            os.rename(p, p2)
        except Exception:
            pass

    if not is_valid_file(p2):
        stats['errors'] += 1
        logging.error(f"Arquivo inválido: {filename}")
        return

    text = extract_text(p2, strat)
    if not text or len(text.strip()) < OCR_THRESHOLD:
        logging.error(f"Não foi possível extrair texto: {filename}")
        stats['errors'] += 1
        return

    rec = build_record(p2, text)
    try:
        inserted_list = save_to_postgres(
            filename, rec['text'], rec['info'],
            model, dim, device, db_name
        )
        stats['processed'] += 1

        if processed_root:
            move_to_processed(p2, processed_root)

        # Quantos chunks foram inseridos no total
        total_chunks = len(inserted_list)
        logging.info(f"→ '{filename}' inseriu {total_chunks} chunks no banco.")

        # Se houve reranking (lista ordenada por 'rerank_score'), mostramos apenas as 3 primeiras scores
        if inserted_list and 'rerank_score' in inserted_list[0]:
            top3 = [f"{r['rerank_score']:.4f}" for r in inserted_list[:3]]
            logging.info(f"    [Rerank] Top 3 scores de '{filename}': {', '.join(top3)}")

    except Exception as e:
        logging.error(f"Erro salvando '{filename}': {e}")
        stats['errors'] += 1
    finally:
        # Forçar remoção de textos e metadados grandes e coletar lixo
        try:
            del text
            del rec
            del inserted_list
            import gc; gc.collect()
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    strat = "ocr"
    model = OLLAMA_EMBEDDING_MODEL
    dim = DIM_MXBAI
    device = "auto"
    allow_tf_cuda = True
    train_dim = dim
    db_name = PG_DB_PDF
    epochs = 1
    batch_size = 1
    eval_steps = EVAL_STEPS
    val_split = VALIDATION_SPLIT
    tokenize_num_proc = TOKENIZE_NUM_PROC
    dataloader_num_workers = DATALOADER_NUM_WORKERS
    stats = {"processed": 0, "errors": 0}
    test_path = ""

    while True:
        clear_screen()
        print("*** Menu Principal ***")
        print(f"1 - Estratégia (atual: {strat})")
        print(f"2 - Embedding  (atual: {model})")
        print(f"3 - Dimensão   (atual: {dim})")
        print(f"4 - Dispositivo (atual: {device})")
        print(f"5 - Banco     (atual: {db_name})")
        print("6 - Processar arquivo")
        print("7 - Processar pasta")
        print("8 - Treinamento")
        print("9 - Testar Modelo")
        print("10 - Sair")
        c = input("> ").strip()

        if c == "10":
            break

        elif c == "1":
            strat = select_strategy(strat)

        elif c == "2":
            model = select_embedding(model)

        elif c == "3":
            dim = select_dimension(dim)
            train_dim = dim

        elif c == "4":
            device = select_device(device)

        elif c == "5":
            db_name = select_database(db_name)

        elif c == "6":
            # Modo “Arquivo”: processa apenas um PDF
            f = input("Arquivo: ").strip()
            if not f:
                print("Nenhum arquivo informado.")
                time.sleep(1)
                continue

            start = time.perf_counter()
            process_file(f, strat, model, dim, device, db_name, stats,
                         os.path.dirname(f))
            dt = time.perf_counter() - start

            print(f"\n→ Tempo gasto: {dt:.2f}s  •  Processados: {stats['processed']}  •  Erros: {stats['errors']}")
            input("ENTER para continuar…")

        elif c == "7":
            # Modo “Pasta”: varre todos os arquivos de dentro de um diretório
            d = input("Pasta: ").strip()
            if not d or not os.path.isdir(d):
                print("Pasta inválida ou não existe.")
                time.sleep(1)
                continue

            # Coleta de todos os PDFs, DOCX e Imagens na pasta recursivamente
            files = []
            for root, dirs, files_ in os.walk(d):
                # ignora subpastas chamadas 'Processado'
                if "Processado" in dirs:
                    dirs.remove("Processado")
                for fname in files_:
                    if fname.lower().endswith(VALID_EXTS):
                        files.append(os.path.join(root, fname))

            total_files = len(files)
            print(f"Total de arquivos encontrados: {total_files}")
            if total_files == 0:
                input("ENTER para continuar…")
                continue

            start = time.perf_counter()

            # tqdm com descrição dinâmica do arquivo atual
            pbar = tqdm(files, unit="arquivo")
            for path in pbar:
                basename = os.path.basename(path)
                # Altera a descrição para mostrar exatamente qual arquivo está sendo processado
                pbar.set_description(
                    f"Processando → {basename} | Strat: {strat} | Emb: {model} | Dim: {dim} | Dev: {device}"
                )
                process_file(path, strat, model, dim, device, db_name, stats, d)
                pbar.set_postfix({"P": stats['processed'], "E": stats['errors']})
                # Coleta lixo após cada arquivo
                try:
                    import gc; gc.collect()
                except Exception:
                    pass

            pbar.close()
            dt = time.perf_counter() - start

            print(f"\n=== Resumo final ===")
            print(f"  Processados: {stats['processed']}  •  Erros: {stats['errors']}  •  Tempo total: {dt:.2f}s")
            input("ENTER para continuar…")

        elif c == "8":
            (
                train_dim,
                allow_tf_cuda,
                epochs,
                batch_size,
                eval_steps,
                val_split,
                tokenize_num_proc,
                dataloader_num_workers,
            ) = training_type_menu(
                train_dim,
                device,
                allow_tf_cuda,
                epochs,
                batch_size,
                eval_steps,
                val_split,
                tokenize_num_proc,
                dataloader_num_workers,
            )

        elif c == "9":
            test_path, device = model_test_menu(test_path, device)

        else:
            print("Opção inválida.")
            time.sleep(1)

    clear_screen()
    print(f"Processados: {stats['processed']}  •  Erros: {stats['errors']}")

if __name__ == "__main__":
    main()

# pg_storage.py
import os
import logging
import json
import psycopg2
import torch
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except Exception as e:
    logging.error(f"Failed to import nltk: {e}")
    nltk = None

try:
    from adaptive_chunker import hierarchical_chunk_generator, get_sbert_model
except Exception as e:
    logging.error(f"Failed to import adaptive_chunker: {e}")
    hierarchical_chunk_generator = None  # type: ignore
    get_sbert_model = None  # type: ignore
from sentence_transformers import CrossEncoder
try:
    from question_generation import pipeline as qg_pipeline
    _QG_AVAILABLE = True
except Exception as e:
    logging.error(f"Falha ao importar question_generation: {e}")
    qg_pipeline = None
    _QG_AVAILABLE = False

from transformers import pipeline as hf_pipeline
from config import (
    PG_HOST,
    PG_PORT,
    PG_USER,
    PG_PASSWORD,
    PG_DB_PDF,
    QG_MODEL,
    QA_MODEL,
    MAX_SEQ_LENGTH,
    QA_EXPLICIT_PROMPT,
)
from metrics import record_metrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


_CE_CACHE: dict = {}
_QG_PIPELINE = None
_QA_PIPELINE = None
_T2T_PIPELINE = None
_QA_MODEL = None
_QA_TOKENIZER = None

def get_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    """Retorna CrossEncoder em cache para o dispositivo escolhido."""
    key = (model_name, device)
    if key not in _CE_CACHE:
        try:
            logging.info(f"Carregando CrossEncoder '{model_name}' em {device}…")
            _CE_CACHE[key] = CrossEncoder(model_name, device=device)
            logging.info(f"CrossEncoder '{model_name}' carregado com sucesso em {device}.")
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device != "cpu":
                logging.error(
                    f"Falha ao carregar CrossEncoder '{model_name}' em {device} devido a falta de memória: {e}; usando CPU"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cpu_key = (model_name, "cpu")
                if cpu_key not in _CE_CACHE:
                    _CE_CACHE[cpu_key] = CrossEncoder(model_name, device="cpu")
                _CE_CACHE[key] = _CE_CACHE[cpu_key]
                logging.info(
                    f"CrossEncoder '{model_name}' carregado com sucesso em CPU (fallback)."
                )
            else:
                logging.error(
                    f"Falha ao carregar CrossEncoder '{model_name}' em {device}: {e}"
                )
                raise
        except Exception as e:
            logging.error(f"Falha ao carregar CrossEncoder '{model_name}' em {device}: {e}")
            raise
    return _CE_CACHE[key]

def generate_embedding(text: str, model_name: str, dim: int, device: str) -> list[float]:
    """Gera embedding no dispositivo escolhido com fallback para CPU."""
    try:
        model = get_sbert_model(model_name, device=device)
        # Garante modo inference (sem gradiente)
        with torch.no_grad():
            emb = model.encode(text, convert_to_numpy=True)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            logging.warning("CUDA OOM – tentando em CPU")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = get_sbert_model(model_name, device="cpu")
            with torch.no_grad():
                emb = model.encode(text, convert_to_numpy=True)
        else:
            logging.error(f"Erro embed genérico: {e}")
            return [0.0] * dim
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return [0.0] * dim

    vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    # Ajusta comprimento para a dimensão correta
    if len(vec) < dim:
        vec += [0.0] * (dim - len(vec))
    elif len(vec) > dim:
        vec = vec[:dim]

    # Limpa cache da GPU (precaução)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return vec


def generate_qa(text: str) -> tuple[str, str]:
    """Gera um par (pergunta, resposta) a partir do texto.

    Retorna ("", "") imediatamente quando o texto possui menos de 50
    caracteres úteis. Caso a pipeline produza pergunta ou resposta vazia,
    o aviso de log inclui o output cru para facilitar a depuração.

    Modelos como ``Narrativa/mT5-base-finetuned-tydiQA-question-generation``
    geram melhores perguntas quando recebem um prompt no formato
    ``answer: <resposta> context: <contexto>``. Esta função detecta tal modelo
    e monta automaticamente o prompt usando a primeira sentença como resposta.
    """
    global _QG_PIPELINE, _QA_PIPELINE, _T2T_PIPELINE, _QA_MODEL, _QA_TOKENIZER

    if len(text.strip()) < 50:
        logging.info("Texto muito curto para gerar QA; pulando")
        return "", ""

    if not _QG_AVAILABLE:
        logging.error("Modulo question_generation ausente; pulando geracao de QA")
        return "", ""

    if _QG_PIPELINE is None and "tydiqa-question-generation" not in QG_MODEL.lower():
        try:
            _QG_PIPELINE = qg_pipeline("question-generation", model=QG_MODEL)
            logging.info(f"QG pipeline loaded with {QG_MODEL}")
        except Exception as e:
            logging.error(
                f"Falha ao carregar pipeline de question generation ({QG_MODEL}): {e}"
            )
            return "", ""

    if QA_EXPLICIT_PROMPT:
        if _QA_MODEL is None or _QA_TOKENIZER is None:
            try:
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSeq2SeqLM,
                    AutoModelForCausalLM,
                )
                _QA_TOKENIZER = AutoTokenizer.from_pretrained(QA_MODEL)
                try:
                    _QA_MODEL = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL)
                except Exception:
                    _QA_MODEL = AutoModelForCausalLM.from_pretrained(QA_MODEL)
                if torch.cuda.is_available():
                    _QA_MODEL = _QA_MODEL.to("cuda")
                logging.info(f"QA model loaded with {QA_MODEL}")
            except Exception as e:
                logging.error(
                    f"Falha ao carregar modelo de QA ({QA_MODEL}) para prompt explicito: {e}"
                )
                _QA_MODEL = None
                _QA_TOKENIZER = None
    elif _QA_PIPELINE is None:
        try:
            _QA_PIPELINE = hf_pipeline("question-answering", model=QA_MODEL)
            logging.info(f"QA pipeline loaded with {QA_MODEL}")
        except Exception as e:
            logging.error(
                f"Falha ao carregar pipeline de question answering ({QA_MODEL}): {e}"
            )
            return "", ""

    try:
        if "tydiqa-question-generation" in QG_MODEL.lower():
            if _T2T_PIPELINE is None:
                try:
                    device_id = 0 if torch.cuda.is_available() else -1
                    _T2T_PIPELINE = hf_pipeline(
                        "text2text-generation", model=QG_MODEL, device=device_id
                    )
                except Exception as e:
                    logging.error(f"Failed to load TyDiQA pipeline: {e}")
                    return "", ""
            if nltk is not None:
                try:
                    answer_span = nltk.sent_tokenize(text)[0]
                except Exception:
                    answer_span = text.split(".")[0]
            else:
                answer_span = text.split(".")[0]
            if not answer_span.strip():
                answer_span = " ".join(text.split()[:20])
            prompt = f"answer: {answer_span.strip()} context: {text}"
            try:
                res = _T2T_PIPELINE(prompt, max_length=64, num_beams=4)
            except Exception as e:
                logging.error(f"TyDiQA question generation failed: {e}")
                return "", ""
            raw_questions = res
        else:
            raw_questions = _QG_PIPELINE(text)

        if isinstance(raw_questions, list):
            questions = []
            for q in raw_questions:
                if isinstance(q, str):
                    questions.append(q)
                elif isinstance(q, dict):
                    questions.append(q.get("question", q.get("generated_text", "")))
                else:
                    questions.append(str(q))
        elif isinstance(raw_questions, str):
            questions = [raw_questions]
        else:
            questions = []

        if not questions:
            logging.warning(
                "QG pipeline produced no questions; attempting fallback generation"
            )
            if _T2T_PIPELINE is None:
                try:
                    device_id = 0 if torch.cuda.is_available() else -1
                    _T2T_PIPELINE = hf_pipeline(
                        "text2text-generation", model=QG_MODEL, device=device_id
                    )
                except Exception as e:
                    logging.error(f"Failed to load fallback pipeline: {e}")
                    return "", ""
            try:
                hl_text = text
                if nltk is not None:
                    try:
                        first_sent = nltk.sent_tokenize(text)[0]
                        hl_text = text.replace(first_sent, f"<hl> {first_sent} <hl>", 1)
                    except Exception as e:
                        logging.error(f"Sentence tokenization failed: {e}")
                res = _T2T_PIPELINE(hl_text, max_length=64, num_beams=4)
                if isinstance(res, list) and res:
                    first = res[0]
                    if isinstance(first, dict):
                        questions = [first.get("generated_text", "").strip()]
                    else:
                        questions = [str(first).strip()]
            except Exception as e:
                logging.error(f"Fallback question generation failed: {e}")
                return "", ""

        question = questions[0].strip() if questions else ""
        
        answer = ""
        qa_res = None
        if question:
            use_pipeline = True
            if QA_EXPLICIT_PROMPT and _QA_MODEL is not None and _QA_TOKENIZER is not None:
                try:
                    input_text = f"question: {question}  context: {text}"
                    data = _QA_TOKENIZER(input_text, return_tensors="pt")
                    data = {k: v.to(_QA_MODEL.device) for k, v in data.items()}
                    out_ids = _QA_MODEL.generate(**data)
                    answer = _QA_TOKENIZER.decode(out_ids[0], skip_special_tokens=True).strip()
                    qa_res = {"answer": answer}
                    use_pipeline = False
                except Exception as e:
                    logging.error(
                        f"QA explicito falhou: {e}; tentando pipeline padrao"
                    )
            if use_pipeline:
                if _QA_PIPELINE is None:
                    try:
                        _QA_PIPELINE = hf_pipeline("question-answering", model=QA_MODEL)
                        logging.info(f"QA pipeline loaded with {QA_MODEL}")
                    except Exception as e:
                        logging.error(
                            f"Falha ao carregar pipeline de question answering ({QA_MODEL}): {e}"
                        )
                        return question, ""

                try:
                    tok_len = len(_QA_PIPELINE.tokenizer.tokenize(text))
                except Exception:
                    tok_len = 0

                try:
                    q_len = len(_QA_PIPELINE.tokenizer.tokenize(question))
                except Exception:
                    q_len = 0

                try:
                    model_max_len = int(_QA_PIPELINE.tokenizer.model_max_length)
                except Exception:
                    model_max_len = MAX_SEQ_LENGTH

                max_seq = min(MAX_SEQ_LENGTH or model_max_len, model_max_len)

                try:
                    specials = _QA_PIPELINE.tokenizer.num_special_tokens_to_add(pair=True)
                except Exception:
                    specials = 0

                max_len = max_seq - specials

                doc_stride = max(1, min(64, tok_len - 1))
                available = max_len - q_len
                if available > 1:
                    doc_stride = min(doc_stride, available - 1)
                else:
                    doc_stride = 1

                logging.debug(f"QA doc_stride={doc_stride} max_len={max_len}")

                kwargs = {"doc_stride": doc_stride, "max_seq_len": max_seq}
                qa_res = _QA_PIPELINE(question=question, context=text, **kwargs)
                if isinstance(qa_res, dict):
                    answer = qa_res.get("answer", "")

        if not question or not answer:
            logging.warning(
                "Pergunta ou resposta vazia gerada em generate_qa - "
                f"QG: {raw_questions!r}, QA: {qa_res!r}"
            )

        return question, answer
    except Exception as e:
        logging.error(f"Erro ao gerar pergunta e resposta: {e}")
        return "", ""


@record_metrics
def save_to_postgres(filename: str,
                     text: str,
                     metadata: dict,
                     embedding_model: str,
                     embedding_dim: int,
                     device: str,
                     db_name: str = PG_DB_PDF) -> list[dict]:
    """
    Insere no PostgreSQL cada chunk gerado em streaming pelo hierarchical_chunk_generator.
    Retorna uma lista de dicionários contendo:
      - 'id': id gerado pelo banco para cada chunk,
      - 'content': texto do chunk,
      - 'metadata': metadados JSONB originais + __parent e __chunk_index.
    Após inserir todos os chunks, se houver chave '__query' em metadata, executa re-ranking
    com CrossEncoder e adiciona o campo 'rerank_score' em cada dicionário antes de ordenar.
    O parâmetro ``db_name`` permite escolher qual banco de dados será utilizado,
    padrão ``PG_DB_PDF``.
    """
    conn = None
    inserted = []

    if device == "auto":
        device_use = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "gpu":
        if torch.cuda.is_available():
            device_use = "cuda"
        else:
            logging.warning("GPU selecionada, mas não disponível. Usando CPU.")
            device_use = "cpu"
    else:
        device_use = "cpu"

    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=db_name,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cur = conn.cursor()

        table = f"public.documents_{embedding_dim}"

        # Inserção em streaming: consome o gerador de chunks
        for idx, chunk in enumerate(hierarchical_chunk_generator(text, metadata, embedding_model, device_use)):
            clean = chunk.replace("\x00", "")
            emb = generate_embedding(clean, embedding_model, embedding_dim, device_use)
            question, answer = generate_qa(clean)

            if not question or not answer:
                logging.warning(
                    f"QA vazio no chunk {idx} do arquivo '{filename}'"
                )

            # Metadata mantém todas as chaves originais + __parent e __chunk_index
            rec = {**metadata, "__parent": filename, "__chunk_index": idx,
                   "question": question, "answer": answer}

            cur.execute(
                f"INSERT INTO {table} (content, metadata, embedding, question, answer) "
                f"VALUES (%s, %s::jsonb, %s, %s, %s) RETURNING id",
                (clean, json.dumps(rec, ensure_ascii=False), emb, question, answer)
            )
            doc_id = cur.fetchone()[0]
            inserted.append({'id': doc_id, 'content': clean, 'metadata': rec,
                             'question': question, 'answer': answer})

            # Limpeza imediata de objetos pesados
            del clean
            del emb
            del rec
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        conn.commit()

        # — Re‐ranking com CrossEncoder se existir __query —
        query = metadata.get('__query', '')
        if query:
            ce = get_cross_encoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device_use)
            pairs = [(query, r['content']) for r in inserted]
            scores = ce.predict(pairs)
            for r, s in zip(inserted, scores):
                r['rerank_score'] = float(s)
            # Ordena pela pontuação de re-ranking (maior para menor)
            inserted.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)

            # Limpa também cache do CrossEncoder
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return inserted

    except Exception as e:
        logging.error(f"Erro saving to Postgres: {e}")
        if conn:
            conn.rollback()
        raise

    finally:
        if conn:
            conn.close()

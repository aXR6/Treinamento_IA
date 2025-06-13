# 🧠 Treinamento e Indexação de Documentos

> **Este repositório é um _fork_ de [aXR6/Extrair_PDF_Agent_IA](https://github.com/aXR6/Extrair_PDF_Agent_IA).**  
> Mantém as capacidades de extração e busca em PDFs/DOCX, com foco principal no **treinamento de modelos da Hugging Face** usando textos armazenados no PostgreSQL.

---

## 🚀 Começando

### Instalação

Certifique-se de ter o Python 3.9 ou superior instalado.

Clone o repositório e utilize o script `init-env.sh` para criar e ativar o
ambiente virtual automaticamente:

```bash
git clone https://github.com/aXR6/Treinamento_IA
cd Treinamento_IA
source init-env.sh
```

O script cria o diretório `.venv`, instala todas as dependências de
`requirements.txt` e deixa o ambiente pronto para uso.

Configure o arquivo `.env` (veja `exemplo.env`) e execute:

```bash
python3 main.py
```

### Variáveis de Ambiente

O arquivo `.env` possibilita ajustar diversos parâmetros do projeto:

- **Modelos de embedding**: `OLLAMA_EMBEDDING_MODEL`, `SERAFIM_EMBEDDING_MODEL`,
  `MINILM_L6_V2`, `MINILM_L12_V2` e `MPNET_EMBEDDING_MODEL`. Defina
  `SBERT_MODEL_NAME` para escolher qual será o padrão.
- **Dimensões**: `DIM_MXBAI`, `DIM_SERAFIM`, `DIM_MINILM_L6`, `DIM_MINIL12` e
  `DIM_MPNET` indicam o tamanho dos vetores em `documents_<dim>`.
- **OCR**: `OCR_LANGUAGES`, `TESSERACT_CONFIG`, `OCR_THRESHOLD` e
  `PDF2IMAGE_TIMEOUT` controlam a extração de texto via OCR.
- **Chunking**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SLIDING_WINDOW_OVERLAP_RATIO`,
  `MAX_SEQ_LENGTH` e `SEPARATORS` determinam como os textos são divididos antes
  da geração dos embeddings. Caso `MAX_SEQ_LENGTH` não seja definido, o padrão
  é `128`.
- **Treinamento**: `TRAINING_MODEL_NAME`, `EVAL_STEPS` e `VALIDATION_SPLIT`
  personalizam o fine-tuning de modelos da Hugging Face.
- **Perguntas e Respostas**: `QG_MODEL` e `QA_MODEL` definem os modelos
  usados para gerar perguntas e respostas (padrão `valhalla/t5-base-qa-qg-hl`).
  Para textos em português, experimente
  `pierreguillou/t5-base-qa-qg-hl-portuguese-squad_v1`.
  **Recomenda-se que `CHUNK_SIZE` seja de no máximo 512 ao gerar perguntas e respostas.**
  A função `generate_qa` limita automaticamente `doc_stride` para nunca exceder o
  tamanho máximo suportado pelo tokenizer. Use `QA_EXPLICIT_PROMPT` para gerar
  respostas com `model.generate` a partir de um prompt "question: ... context:
  ...". Modelos generativos como `Narrativa/mT5-base-finetuned-tydiQA-xqa` devem
  ser carregados com `AutoModelForCausalLM` para que `generate` funcione. Exemplo:

  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  tok = AutoTokenizer.from_pretrained("Narrativa/mT5-base-finetuned-tydiQA-xqa")
  model = AutoModelForCausalLM.from_pretrained(
      "Narrativa/mT5-base-finetuned-tydiQA-xqa")
  prompt = f"question: {question_text} context: {context_text}"
  answer = tok.decode(model.generate(**tok(prompt, return_tensors='pt'))[0])
  ```
- **Outros**: `CSV_FULL` e `CSV_INCR` podem apontar para arquivos CSV locais de
  vulnerabilidades (opcional).

#### TyDiQA

Defina `QG_MODEL` como
`Narrativa/mT5-base-finetuned-tydiQA-question-generation` e `QA_MODEL` como
`Narrativa/mT5-base-finetuned-tydiQA-xqa`. Para obter os melhores resultados,
esse gerador de perguntas espera um prompt no formato
`answer: <resposta> context: <contexto>`.
A cobertura inclui os idiomas:
(Afrikaans, Albanian, Amharic, Arabic, Armenian, Azerbaijani, Basque, Belarusian, Bengali, Bulgarian, Burmese, Catalan, Cebuano, Chichewa, Chinese, Corsican, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hausa, Hawaiian, Hebrew, Hindi, Hmong, Hungarian, Icelandic, Igbo, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish, Kyrgyz, Lao, Latin, Latvian, Lithuanian, Luxembourgish, Macedonian, Malagasy, Malay, Malayalam, Maltese, Maori, Marathi, Mongolian, Nepali, Norwegian, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Samoan, Scottish Gaelic, Serbian, Shona, Sindhi, Sinhala, Slovak, Slovenian, Somali, Sotho, Spanish, Sundanese, Swahili, Swedish, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese, Welsh, West Frisian, Xhosa, Yiddish, Yoruba, Zulu.), incluindo suporte para português.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("Narrativa/mT5-base-finetuned-tydiQA-question-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("Narrativa/mT5-base-finetuned-tydiQA-question-generation")
prompt = f"answer: {answer_text} context: {context_text}"
question = tok.decode(model.generate(**tok(prompt, return_tensors='pt'))[0])
```

O prompt precisa dos prefixos `answer:` e `context:`.

#### Exemplo simplificado de `.env`

```env
PG_HOST=192.168.3.32
PG_PORT=5432
PG_USER=vector_store
PG_PASSWORD=senha
PG_DB_PDF=vector_store_pdf
PG_DB_QA=vector_store_pdf_qs

TRAINING_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
EVAL_STEPS=500
VALIDATION_SPLIT=0.1
QG_MODEL=valhalla/t5-base-qa-qg-hl
QA_MODEL=${QG_MODEL}
# Ativa prompt explicito no QA (opcional)
# QA_EXPLICIT_PROMPT=1
# QG_MODEL=Narrativa/mT5-base-finetuned-tydiQA-question-generation
# QA_MODEL=Narrativa/mT5-base-finetuned-tydiQA-xqa
```

### Estrutura do Projeto

```
├─ main.py           # CLI para indexação e treinamento
├─ training.py       # Funções de fine-tuning de modelos
├─ init-env.sh       # Criação/ativação do virtualenv
├─ BD_PostgreSQL/    # Scripts SQL da estrutura do banco
├─ Srv/              # Microserviço FastAPI para embeddings
```

---

## 🏋️ Treinamento de Modelos

1. Defina o modelo desejado em `TRAINING_MODEL_NAME` no `.env`. Se não definido, será usado `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
2. Processe seus documentos normalmente (opções 1 a 7 do menu) para popular a tabela `public.documents_<dim>`.
3. Escolha a dimensão (opção 3) e o dispositivo (opção 4).
4. Acesse **8 - Treinamento** ou **9 - Treinamento QA**. Nos submenus você pode:
   - Definir a tabela de origem (opção 3).
   - Ajustar épocas, batch size, passos de avaliação e porcentagem de validação.
   - Ativar ou não a detecção automática de GPU.
   - Por fim, escolha **1 - Treinar modelo**.
5. O resultado é salvo em uma pasta `MODELNAME_finetuned_<dim>` e o melhor modelo em `best_model`.

### Dependências

- `transformers`
- `sentence-transformers`
- `datasets`
- `accelerate`
- `torch`
- `psycopg2-binary`
- `question-generation`

`pg_storage.py` tenta importar o pacote `question_generation`. Caso a
dependência não esteja disponível, uma mensagem de erro é registrada e a função
`generate_qa` retorna pares vazios.

Todas as dependências estão listadas em `requirements.txt`.

---

## ⚙️ Funcionalidades

- **Extração de Texto:** PDF, DOCX e imagens com múltiplas estratégias
  (PyPDFLoader, PDFMiner, Unstructured, OCR, etc.) e reparo automático de PDFs.
- **Chunking Inteligente:** algoritmo hierárquico que agrupa parágrafos,
  expande queries com WordNet e usa sliding window quando necessário.
- **Embeddings e Indexação:** geração de vetores com SBERT e inserção em
  streaming no PostgreSQL (pgvector), permitindo busca híbrida (RAG).
- **Pares de Pergunta/Resposta:** as tabelas `documents_<dim>` agora incluem as
  colunas `question` e `answer` para armazenar contextos já respondidos e
  otimizar workflows de RAG. Se a geração falhar, um *warning* indica o índice do
  chunk e o arquivo correspondente.
- **Re-ranking e Métricas:** Cross-Encoder para ordenar resultados e servidor
  Prometheus embutido para monitorar consultas.
- **CLI Interativo:** escolha de estratégia, modelo, dimensão, dispositivo e
  parâmetros de treinamento, com barra de progresso e logs detalhados.

---

## 📄 Licença

Consulte o arquivo [LICENSE](./LICENSE) para mais informações.

---

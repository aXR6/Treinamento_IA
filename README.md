# 🧠 Treinamento e Indexação de Documentos

Este repositório é um **fork de [aXR6/Extrair_PDF_Agent_IA](https://github.com/aXR6/Extrair_PDF_Agent_IA)** e mantém suas capacidades de extração e busca em PDFs/DOCX. O foco principal, entretanto, é o **treinamento de modelos da Hugging Face** usando textos armazenados no PostgreSQL.

## Treinamento de Modelos

1. Defina o modelo desejado em `TRAINING_MODEL_NAME` dentro do `.env`. Caso não indique nenhum valor, o script usa `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
2. Processe seus documentos normalmente (opções 1 a 6 do menu) para popular a tabela `public.documents_<dim>`.
3. Escolha a dimensão (opção 3) e o dispositivo (opção 4). Se desejar que o transformers detecte a GPU automaticamente, habilite a opção 8.
4. Selecione **7 - Treinar modelo**. O programa coleta os textos da tabela indicada, monta um dataset com `datasets` e realiza o ajuste fino via `transformers.Trainer`.
5. O resultado é salvo em uma pasta `MODELNAME_finetuned_<dim>`.

### Dependências de Treinamento

O pipeline utiliza `transformers`, `sentence-transformers`, `datasets`, `accelerate`, `torch` e `psycopg2-binary`. Todos os pacotes estão listados em `requirements.txt`.

---

## Demais Funcionalidades

Mesmo priorizando o treinamento, o projeto continua oferecendo:

- **Extração de Texto:** PDF, DOCX e imagens com múltiplas estratégias (PyPDFLoader, PDFMiner, Unstructured, OCR, etc.).
- **Chunking Inteligente:** filtros de parágrafo, identificação de headings, sliding window e fallback automático.
- **Embeddings e Indexação:** geração de vetores via SBERT, armazenamento no PostgreSQL com pgvector e busca híbrida (RAG).
- **Re-ranking e Métricas:** Cross-Encoder para melhor precisão e monitoramento via Prometheus.
- **CLI Interativo:** escolha de estratégia, modelo, dimensão, dispositivo e treinamento, com barra de progresso e log detalhado.

---

## Instalação Rápida

```bash
git clone https://github.com/seu_usuario/seu_projeto.git
cd seu_projeto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure o arquivo `.env` (veja `exemplo.env`) e execute:

```bash
python3 main.py
```

---

## Exemplo Simplificado de `.env`

```env
PG_HOST=192.168.3.32
PG_PORT=5432
PG_USER=vector_store
PG_PASSWORD=senha
PG_DATABASE=vector_store

TRAINING_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

Complete o arquivo com as demais variáveis descritas em `exemplo.env` para controlar dimensões, modelos de embedding, parâmetros de OCR e chunking.


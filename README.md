# 🧠 Treinamento e Indexação de Documentos

> **Este repositório é um _fork_ de [aXR6/Extrair_PDF_Agent_IA](https://github.com/aXR6/Extrair_PDF_Agent_IA).**  
> Mantém as capacidades de extração e busca em PDFs/DOCX, com foco principal no **treinamento de modelos da Hugging Face** usando textos armazenados no PostgreSQL.

---

## 🚀 Começando

### Instalação

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

#### Exemplo simplificado de `.env`

```env
PG_HOST=192.168.3.32
PG_PORT=5432
PG_USER=vector_store
PG_PASSWORD=senha
PG_DATABASE=vector_store

TRAINING_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
EVAL_STEPS=500
VALIDATION_SPLIT=0.1
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
2. Processe seus documentos normalmente (opções 1 a 6 do menu) para popular a tabela `public.documents_<dim>`.
3. Escolha a dimensão (opção 3) e o dispositivo (opção 4).
4. Acesse **7 - Treinamento**. No submenu você pode:
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

Todas as dependências estão listadas em `requirements.txt`.

---

## ⚙️ Funcionalidades

- **Extração de Texto:** PDF, DOCX e imagens com múltiplas estratégias (PyPDFLoader, PDFMiner, Unstructured, OCR, etc.).
- **Chunking Inteligente:** Filtros de parágrafo, identificação de headings, sliding window e fallback automático.
- **Embeddings e Indexação:** Geração de vetores via SBERT, armazenamento no PostgreSQL com pgvector e busca híbrida (RAG).
- **Re-ranking e Métricas:** Cross-Encoder para melhor precisão e monitoramento via Prometheus.
- **CLI Interativo:** Escolha de estratégia, modelo, dimensão, dispositivo e treinamento, com barra de progresso e logs detalhados.

---

## 📄 Licença

Consulte o arquivo [LICENSE](./LICENSE) para mais informações.

---

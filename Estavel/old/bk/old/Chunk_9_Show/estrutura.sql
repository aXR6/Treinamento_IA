-- Estrutura do banco de dados para armazenar documentos e embeddings
-- Enable the pgvector extension to work with embedding vectors
create extension vector;

-- Create a table to store your documents
create table documents (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1024) -- 1024 works for OpenAI embeddings, change if needed
);

-- Create a function to search for documents
create function match_documents (
  query_embedding vector(1024),
  match_count int default null,
  filter jsonb DEFAULT '{}'
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Funções para criar e atualizar o índice de texto completo
ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS tsv_full tsvector;
  
CREATE OR REPLACE FUNCTION update_tsv_full() RETURNS trigger AS $$
BEGIN
  NEW.tsv_full :=
    setweight(to_tsvector('simple', coalesce(NEW.content, '')), 'A') ||
    setweight(to_tsvector('simple', COALESCE(NEW.metadata::text, '')), 'B');
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsv_full_trigger ON documents;
CREATE TRIGGER tsv_full_trigger
  BEFORE INSERT OR UPDATE ON documents
  FOR EACH ROW EXECUTE FUNCTION update_tsv_full();


-- Vetores: IVFFlat ou HNSW (ajuste nlist/pq)
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Text search
CREATE INDEX ON documents USING gin (tsv_full);


CREATE OR REPLACE FUNCTION match_documents_hybrid(
  query_embedding VECTOR(1024),
  query_text      TEXT,
  match_count     INT     DEFAULT 5,
  filter          JSONB   DEFAULT '{}',
  weight_vec      FLOAT   DEFAULT 0.5,
  weight_lex      FLOAT   DEFAULT 0.5,
  min_score       FLOAT   DEFAULT 0.1
) RETURNS TABLE (
  id     BIGINT,
  content TEXT,
  metadata JSONB,
  score   FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  WITH raw AS (
    SELECT
      metadata ->> '__parent'                           AS parent,
      id, content, metadata,
      -- Similaridade vetorial (cosseno invertido)
      (1 - (embedding <=> query_embedding))             AS sim,
      -- Ranking léxico com sintaxe de busca web
      ts_rank(tsv_full, websearch_to_tsquery(query_text)) AS lex_rank
    FROM documents
    WHERE metadata @> filter
      AND (query_text IS NULL OR tsv_full @@ websearch_to_tsquery(query_text))
  ),
  scored AS (
    SELECT
      parent,
      id, content, metadata,
      -- Combinação ponderada e fallback
      COALESCE(
        weight_vec * sim + weight_lex * lex_rank,
        sim
      )                                                AS score
    FROM raw
  ),
  best_parent AS (
    SELECT parent
    FROM scored
    ORDER BY score DESC
    LIMIT 1
  )
  SELECT
    id, content, metadata, score
  FROM scored
  WHERE parent = (SELECT parent FROM best_parent)
    AND score >= min_score
  ORDER BY score DESC
  LIMIT match_count;
END;
$$;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE plans (
    id                      text PRIMARY KEY,
    name                    text NOT NULL,
    plan_year               int  NOT NULL,
    deductible              numeric(12,2) NOT NULL CHECK (deductible >= 0),
    coinsurance_member_pct  numeric(5,2)  NOT NULL CHECK (coinsurance_member_pct BETWEEN 0 AND 100),
    oop_max                 numeric(12,2) NOT NULL CHECK (oop_max >= 0)
);

CREATE TABLE members (
    id              text PRIMARY KEY,
    plan_id         text NOT NULL REFERENCES plans(id),
    active          boolean NOT NULL DEFAULT true,
    coverage_start  date NOT NULL,
    coverage_end    date
);

CREATE TABLE accumulators (
    member_id        text NOT NULL REFERENCES members(id),
    plan_year        int  NOT NULL,
    deductible_paid  numeric(12,2) NOT NULL DEFAULT 0,
    oop_paid         numeric(12,2) NOT NULL DEFAULT 0,
    PRIMARY KEY (member_id, plan_year)
);

CREATE TABLE services (
    code                 text PRIMARY KEY,
    name                 text NOT NULL,
    aliases              text[] NOT NULL DEFAULT '{}',
    category             text NOT NULL,
    requires_prior_auth  boolean NOT NULL DEFAULT false,
    typical_charge       numeric(12,2) NOT NULL
);

CREATE TABLE claims (
    id             text PRIMARY KEY,
    member_id      text NOT NULL REFERENCES members(id),
    service_code   text NOT NULL REFERENCES services(code),
    service_date   date NOT NULL,
    provider       text,
    billed_amount  numeric(12,2) NOT NULL,
    status         text NOT NULL CHECK (status IN ('submitted','in_review','approved','denied','appealed')),
    denial_code    text,
    denial_reason  text,
    member_paid    numeric(12,2) NOT NULL DEFAULT 0,
    plan_paid      numeric(12,2) NOT NULL DEFAULT 0
);
CREATE INDEX claims_member_date_idx ON claims (member_id, service_date DESC);

CREATE TABLE prior_authorizations (
    id            text PRIMARY KEY,
    member_id     text NOT NULL REFERENCES members(id),
    service_code  text NOT NULL REFERENCES services(code),
    status        text NOT NULL CHECK (status IN ('pending','approved','denied','expired')),
    requested_at  timestamptz NOT NULL,
    decided_at    timestamptz,
    valid_until   date
);
CREATE INDEX prior_auth_member_service_idx ON prior_authorizations (member_id, service_code, requested_at DESC);

-- Embedding dimension must match CA_EMBEDDING_DIM. Changing models = new migration + re-ingest.
CREATE TABLE document_chunks (
    source        text NOT NULL,
    chunk_index   int  NOT NULL,
    doc_type      text NOT NULL,
    content       text NOT NULL,
    content_hash  text NOT NULL,
    embedding     vector(384) NOT NULL,
    updated_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, chunk_index)
);
CREATE INDEX document_chunks_doc_type_idx ON document_chunks (doc_type);
CREATE INDEX document_chunks_embedding_idx ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Append-only access log. Grant the app role INSERT only; ship to WORM storage for retention.
CREATE TABLE audit_log (
    id          bigserial PRIMARY KEY,
    at          timestamptz NOT NULL DEFAULT now(),
    actor_id    text NOT NULL,
    member_id   text,
    action      text NOT NULL,
    request_id  text
);
CREATE INDEX audit_log_member_idx ON audit_log (member_id, at DESC);

-- Every answered case, for audit, QA review and building eval sets. Contains PHI: apply retention.
CREATE TABLE case_runs (
    id                 uuid PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    member_id          text NOT NULL,
    actor_id           text NOT NULL,
    question           text NOT NULL,
    outcome            text NOT NULL,
    routed_to          text[] NOT NULL,
    facts              jsonb NOT NULL,
    answer             text NOT NULL,
    latency_ms         int NOT NULL,
    prompt_tokens      int,
    completion_tokens  int,
    llm_calls          int
);
CREATE INDEX case_runs_created_idx ON case_runs (created_at DESC);
CREATE INDEX case_runs_outcome_idx ON case_runs (outcome, created_at DESC);

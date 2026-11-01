-- Staff work queue for cases paused by the LangGraph human_review interrupt.
-- The graph checkpoint holds the full case; this table is the index staff browse.
CREATE TABLE review_queue (
    case_id      uuid PRIMARY KEY,
    created_at   timestamptz NOT NULL DEFAULT now(),
    member_id    text NOT NULL,
    reason       text NOT NULL,
    question     text NOT NULL,
    draft        text,
    status       text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done')),
    reviewer_id  text,
    decision     text CHECK (decision IN ('approve', 'edit', 'reject')),
    decided_at   timestamptz
);
CREATE INDEX review_queue_pending_idx ON review_queue (status, created_at);

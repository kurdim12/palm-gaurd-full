-- Palm Guard schema for Supabase / PostgreSQL.
-- Apply with: psql "$SUPABASE_DB_URL" -f app/db/schema.sql
-- or paste into the Supabase SQL editor.

create extension if not exists "pgcrypto";

-- Monitored trees.
create table if not exists trees (
    id                text primary key,
    name_ar           text not null default '',
    name_en           text not null default '',
    lat               double precision not null,
    lon               double precision not null,
    status            text not null default 'clean'
                          check (status in ('clean','suspect','infested','treated')),
    infested_streak   integer not null default 0,
    last_detection_at timestamptz,
    updated_at        timestamptz not null default now()
);

-- Raw edge detections (append-only).
create table if not exists detections (
    id            uuid primary key default gen_random_uuid(),
    device_id     text not null,
    tree_id       text not null references trees(id) on delete cascade,
    label         text not null check (label in ('clean','infested')),
    confidence    double precision not null check (confidence between 0 and 1),
    captured_at   timestamptz not null,
    received_at   timestamptz not null default now(),
    audio_url     text,
    model_version text
);

create index if not exists detections_tree_captured_idx
    on detections (tree_id, captured_at desc);

-- Infestation alerts.
create table if not exists alerts (
    id              uuid primary key default gen_random_uuid(),
    tree_id         text not null references trees(id) on delete cascade,
    created_at      timestamptz not null default now(),
    acknowledged    boolean not null default false,
    acknowledged_at timestamptz,
    message_ar      text not null default '',
    message_en      text not null default ''
);

create index if not exists alerts_open_created_idx
    on alerts (acknowledged, created_at desc);

-- Row Level Security: enable and lock down. The API uses the service key
-- (bypasses RLS); add explicit anon/auth policies before exposing PostgREST.
alter table trees      enable row level security;
alter table detections enable row level security;
alter table alerts     enable row level security;

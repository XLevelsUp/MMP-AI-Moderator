-- Migration 1 of 2: moderation_schema
-- Adds the status columns, the moderation.decisions table and the moderation_service role.
-- Safe to run early: existing and new rows default to 'approved', so nothing users see changes.
-- After running, an admin gives the role a login:
--   alter role moderation_service login password '<strong password>';
-- Undo with rollback/20261104100000_moderation_schema_down.sql (this deletes moderation.decisions).

begin;

-- ===== 1. Columns on existing tables =====
-- DEFAULT 'approved' keeps every existing row visible. Those rows were never checked by this service.
alter table social.posts
  add column moderation_status varchar(20) not null default 'approved'
    constraint posts_moderation_status_check
    check (moderation_status in ('pending', 'in_review', 'approved', 'rejected'));

alter table social.post_media
  add column moderation_status varchar(20) not null default 'approved'
    constraint post_media_moderation_status_check
    check (moderation_status in ('pending', 'in_review', 'approved', 'rejected')),
  add column moderation_rejection_code varchar(10)
    constraint post_media_moderation_rejection_code_check
    check (moderation_rejection_code ~ '^MOD_00[1-9]$');

alter table pets.pet_media
  add column moderation_status varchar(20) not null default 'approved'
    constraint pet_media_moderation_status_check
    check (moderation_status in ('pending', 'in_review', 'approved', 'rejected')),
  add column moderation_rejection_code varchar(10)
    constraint pet_media_moderation_rejection_code_check
    check (moderation_rejection_code ~ '^MOD_00[1-9]$');

alter table pets.pets
  add column profile_photo_moderation_status varchar(20) not null default 'approved'
    constraint pets_profile_photo_moderation_status_check
    check (profile_photo_moderation_status in ('pending', 'in_review', 'approved', 'rejected')),
  add column profile_photo_moderation_rejection_code varchar(10)
    constraint pets_profile_photo_moderation_rejection_code_check
    check (profile_photo_moderation_rejection_code ~ '^MOD_00[1-9]$');

-- ===== 2. The one new table =====
create schema moderation;

create table moderation.decisions (
  id uuid primary key default gen_random_uuid(),         -- the API's content_id

  -- What was checked. No foreign keys, like public.admin_audit_logs: deleting an account or post
  -- must neither erase this record nor be blocked by it.
  account_id uuid not null,                              -- identity.accounts.id
  pet_id uuid,                                           -- pets.pets.id
  media_type varchar(20) not null,                      -- post_media, pet_media, pet_profile_photo
  media_id uuid not null,                               -- the row whose status gets updated
  storage_path text not null,                            -- bucket and path, never a token
  media_sha256 character(64),                            -- null if the file never downloaded
  media_size_bytes bigint,
  content_type varchar(10) not null,
  context varchar(20) not null,

  -- The AI's answer
  decision varchar(20) not null,
  rejection_code varchar(10),
  hard_reject boolean not null default false,
  pet_score numeric(5,2),
  composite_score numeric(5,2),
  dimension_scores jsonb,
  ai_generated boolean,
  ai_score numeric(5,2),
  reason_codes text[] not null default '{}',
  flags text[] not null default '{}',
  species_detected text[] not null default '{}',
  detected_objects jsonb,                                -- debugging only, not in the API answer
  processing_ms integer not null,
  model_version text not null,
  created_at timestamptz not null default now(),

  -- pending = waiting for a moderator. done = nothing left to do: a moderator reviewed it
  -- (reviewed_by is set) or the AI handled it alone (reviewed_by is null).
  review_status varchar(10) not null,
  final_decision varchar(20),
  final_rejection_code varchar(10),
  incorrect_reason_codes text[] not null default '{}',
  missing_reason_codes text[] not null default '{}',
  remarks text,
  reviewed_by uuid,                                      -- identity.accounts.id
  reviewed_at timestamptz,

  -- The breed check: recorded only, it never changes moderation_status. Not built yet: always not_checked.
  declared_breed_id uuid,                                -- pets.breeds.id, sent by the app
  declared_is_cross_breed boolean,
  breed_status varchar(15) not null default 'not_checked',
  breed_match_score numeric(5,2),                        -- how well the photo fits the claimed breed
  detected_breed_id uuid,                                -- pets.breeds.id: what the AI found, when unmatched
  detected_breed_name varchar(150),
  detected_breed_score numeric(5,2),

  constraint decisions_breed_status_check check (breed_status in ('matched', 'unmatched', 'not_checked')),
  constraint decisions_breed_unmatched_check check (
    breed_status <> 'unmatched' or (breed_match_score is not null and detected_breed_id is not null)
  ),

  -- Today every post is one pets.pet_media row with the same id (trg_sync_pet_media_to_post);
  -- social.post_media has 0 rows.
  constraint decisions_context_media_check check (
    (context = 'post' and media_type in ('pet_media', 'post_media'))
    or (context = 'profile' and media_type = 'pet_profile_photo')
  ),
  constraint decisions_content_type_check check (content_type in ('image', 'video')),
  constraint decisions_decision_check check (decision in ('APPROVED', 'HUMAN_REVIEW', 'REJECTED')),
  constraint decisions_rejection_code_check check (rejection_code ~ '^MOD_00[1-9]$'),
  constraint decisions_rejected_has_code_check check ((decision = 'REJECTED') = (rejection_code is not null)),
  constraint decisions_review_status_check check (review_status in ('pending', 'done')),
  constraint decisions_final_decision_check check (final_decision in ('APPROVED', 'REJECTED')),
  constraint decisions_final_rejection_code_check check (final_rejection_code ~ '^MOD_00[1-9]$'),
  constraint decisions_final_rejected_has_code_check
    check ((final_decision is not distinct from 'REJECTED') = (final_rejection_code is not null)),
  -- The moderator's fields are all empty (pending, or handled by the AI) or all filled in on a done row.
  constraint decisions_review_fields_check check (
    (final_decision is null and reviewed_by is null and reviewed_at is null)
    or (review_status = 'done' and final_decision is not null and reviewed_by is not null and reviewed_at is not null)
  ),
  constraint decisions_remarks_length_check check (char_length(remarks) <= 1000)
);

-- The queue: holds pending rows only, so it stays small however many decisions build up.
create index decisions_queue_idx on moderation.decisions (created_at) where review_status = 'pending';
create index decisions_media_idx on moderation.decisions (media_type, media_id, created_at desc);
create index decisions_account_idx on moderation.decisions (account_id, created_at desc);
create index decisions_breed_unmatched_idx on moderation.decisions (created_at) where breed_status = 'unmatched';

-- Not reachable with the app's public keys. Don't add this schema to the exposed Data API schemas.
alter table moderation.decisions enable row level security;

-- ===== 3. Least-privilege role for the moderation service =====
-- service_role has no UPDATE on these tables today, so the service gets its own role.
-- Row-level security is bypassed, but column grants limit it to exactly these columns.
create role moderation_service nologin bypassrls;

grant usage on schema moderation, social, pets, identity, public to moderation_service;
grant select, insert, update on moderation.decisions to moderation_service;

grant select (id), update (moderation_status) on social.posts to moderation_service;
grant select (id, post_id, moderation_status, storage_path), update (moderation_status, moderation_rejection_code) on social.post_media to moderation_service;
grant select (id, storage_path), update (moderation_status, moderation_rejection_code) on pets.pet_media to moderation_service;
grant select (id, profile_photo_path), update (profile_photo_moderation_status, profile_photo_moderation_rejection_code) on pets.pets to moderation_service;

grant select (id, email, is_platform_moderator, is_platform_admin) on identity.accounts to moderation_service;
grant insert on public.admin_audit_logs to moderation_service;

-- ===== 4. The admin panel (service_role): filter, read the AI's results, approve and reject =====
-- The AI's own answer stays read-only: only the review columns and the status columns can change.
grant usage on schema moderation to service_role;
grant select on moderation.decisions to service_role;
grant select on pets.pet_media to service_role;

grant update (review_status, final_decision, final_rejection_code, incorrect_reason_codes,
              missing_reason_codes, remarks, reviewed_by, reviewed_at)
  on moderation.decisions to service_role;
grant update (moderation_status) on social.posts to service_role;
grant update (moderation_status, moderation_rejection_code) on social.post_media, pets.pet_media to service_role;
grant update (profile_photo_moderation_status, profile_photo_moderation_rejection_code) on pets.pets to service_role;

commit;

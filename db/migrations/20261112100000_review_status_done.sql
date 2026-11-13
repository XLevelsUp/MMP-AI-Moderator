-- review_status: only 'pending' or 'done', never empty (decided 12 Nov 2026).
-- For a database that already ran 20261104100000_moderation_schema before that date, such as staging.
-- A new database doesn't need it, but running it there changes nothing.
-- Rows the AI handled alone (approved, bad file, hash-list match) become 'done' with reviewed_by empty.
-- Undo with rollback/20261112100000_review_status_done_down.sql.

begin;

-- The old rule refuses 'done' without a moderator, so it goes first.
alter table moderation.decisions drop constraint if exists decisions_review_done_check;
alter table moderation.decisions drop constraint if exists decisions_review_fields_check;

update moderation.decisions set review_status = 'done' where review_status is null;

alter table moderation.decisions alter column review_status set not null;

-- The moderator's fields are all empty (pending, or handled by the AI) or all filled in on a done row.
alter table moderation.decisions add constraint decisions_review_fields_check check (
  (final_decision is null and reviewed_by is null and reviewed_at is null)
  or (review_status = 'done' and final_decision is not null and reviewed_by is not null and reviewed_at is not null)
);

commit;

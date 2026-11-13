-- Undoes 20261112100000_review_status_done: rows the AI handled alone go back to an empty review_status.

begin;

alter table moderation.decisions drop constraint if exists decisions_review_fields_check;
alter table moderation.decisions alter column review_status drop not null;

update moderation.decisions set review_status = null where review_status = 'done' and reviewed_by is null;

alter table moderation.decisions add constraint decisions_review_done_check check (
  (review_status is not distinct from 'done') = (final_decision is not null and reviewed_by is not null and reviewed_at is not null)
);

commit;

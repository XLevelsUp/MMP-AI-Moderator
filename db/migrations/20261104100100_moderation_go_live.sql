-- Migration 2 of 2: moderation_go_live
-- Run only after migration 1, and only when the app and the engine already handle every new upload.
-- From this moment new uploads stay hidden until the engine approves them.
-- Compare the three policies in step 4 with the live ones first.
-- Not covered: pets.pets rows must stay readable, so the app shows a placeholder until the main photo is approved.
-- Public buckets: anyone holding a file's link can still open it; row-level security hides the row, not the file.
-- Undo with rollback/20261104100100_moderation_go_live_down.sql.

begin;

-- ===== 1. New uploads start hidden =====
alter table social.posts      alter column moderation_status set default 'pending';
alter table social.post_media alter column moderation_status set default 'pending';
alter table pets.pet_media    alter column moderation_status set default 'pending';
alter table pets.pets         alter column profile_photo_moderation_status set default 'pending';

-- ===== 2. Users can't set their own status; changing a file forces a new check =====
create function moderation.protect_post_status() returns trigger
language plpgsql set search_path = '' as $$
begin
  if current_user not in ('authenticated', 'anon') then
    return new;
  end if;
  if tg_op = 'INSERT' then
    new.moderation_status := 'pending';
  else
    new.moderation_status := old.moderation_status;
  end if;
  return new;
end;
$$;

create function moderation.protect_post_media_status() returns trigger
language plpgsql set search_path = '' as $$
begin
  if current_user not in ('authenticated', 'anon') then
    return new;
  end if;
  new.moderation_status := 'pending';
  new.moderation_rejection_code := null;
  return new;
end;
$$;

create function moderation.protect_pet_media_status() returns trigger
language plpgsql set search_path = '' as $$
begin
  if current_user not in ('authenticated', 'anon') then
    return new;
  end if;
  if tg_op = 'INSERT'
     or new.url is distinct from old.url
     or new.storage_path is distinct from old.storage_path then
    new.moderation_status := 'pending';
    new.moderation_rejection_code := null;
  else
    new.moderation_status := old.moderation_status;
    new.moderation_rejection_code := old.moderation_rejection_code;
  end if;
  return new;
end;
$$;

create function moderation.protect_pet_photo_status() returns trigger
language plpgsql set search_path = '' as $$
begin
  if current_user not in ('authenticated', 'anon') then
    return new;
  end if;
  if tg_op = 'INSERT'
     or new.profile_photo_url is distinct from old.profile_photo_url
     or new.profile_photo_path is distinct from old.profile_photo_path then
    new.profile_photo_moderation_status := 'pending';
    new.profile_photo_moderation_rejection_code := null;
  else
    new.profile_photo_moderation_status := old.profile_photo_moderation_status;
    new.profile_photo_moderation_rejection_code := old.profile_photo_moderation_rejection_code;
  end if;
  return new;
end;
$$;

create trigger zz_protect_post_status before insert or update on social.posts
  for each row execute function moderation.protect_post_status();
-- post_media has no update policy, so only inserts need this.
create trigger zz_protect_post_media_status before insert on social.post_media
  for each row execute function moderation.protect_post_media_status();
create trigger zz_protect_pet_media_status before insert or update on pets.pet_media
  for each row execute function moderation.protect_pet_media_status();
create trigger zz_protect_pet_photo_status before insert or update on pets.pets
  for each row execute function moderation.protect_pet_photo_status();

-- ===== 3. A pet_media status always carries to its post, which has the same id =====
-- The service updates only pets.pet_media. SECURITY DEFINER so a re-check forced by a user's file change
-- also holds the post, past the protection above.
create function moderation.sync_pet_media_status_to_post() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  update social.posts set moderation_status = new.moderation_status where id = new.id;
  return null;
end;
$$;

revoke all on function moderation.sync_pet_media_status_to_post() from public;

create trigger trg_sync_pet_media_status_to_post after update on pets.pet_media
  for each row when (old.moderation_status is distinct from new.moderation_status)
  execute function moderation.sync_pet_media_status_to_post();

-- ===== 4. Other users see only approved content; the owner still sees their own =====
-- These are the existing policies with the status condition added. Compare them with the live ones first.
alter policy "Users can view posts from eligible pets" on social.posts
  using (
    (pet_id in (select social.caller_pet_ids()))
    or (deleted_at is null and moderation_status = 'approved' and social.can_view_pet_posts(pet_id))
  );

alter policy "Users can view media on visible posts" on social.post_media
  using (
    exists (
      select 1 from social.posts p
      where p.id = post_media.post_id
        and (
          (p.pet_id in (select social.caller_pet_ids()))
          or (p.deleted_at is null and post_media.moderation_status = 'approved' and social.can_view_pet_posts(p.pet_id))
        )
    )
  );

alter policy pet_media_select_visible on pets.pet_media
  using (
    exists (select 1 from pets.pets p where p.id = pet_media.pet_id)
    and (
      moderation_status = 'approved'
      or pet_id in (
        select p.id from pets.pets p
        where p.owner_account_id in (select a.id from identity.accounts a where a.auth_user_id = auth.uid())
      )
    )
  );

commit;

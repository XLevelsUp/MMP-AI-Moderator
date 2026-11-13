-- Undoes migration 2: new uploads are visible again, and the read rules return to what they were.
-- Existing statuses stay in their columns; nothing is deleted.

begin;

alter policy "Users can view posts from eligible pets" on social.posts
  using (
    (pet_id in (select social.caller_pet_ids()))
    or ((deleted_at is null) and social.can_view_pet_posts(pet_id))
  );

alter policy "Users can view media on visible posts" on social.post_media
  using (
    exists (
      select 1 from social.posts p
      where p.id = post_media.post_id
        and (
          (p.pet_id in (select social.caller_pet_ids()))
          or ((p.deleted_at is null) and social.can_view_pet_posts(p.pet_id))
        )
    )
  );

alter policy pet_media_select_visible on pets.pet_media
  using (exists (select 1 from pets.pets p where p.id = pet_media.pet_id));

drop trigger trg_sync_pet_media_status_to_post on pets.pet_media;
drop trigger zz_protect_post_status on social.posts;
drop trigger zz_protect_post_media_status on social.post_media;
drop trigger zz_protect_pet_media_status on pets.pet_media;
drop trigger zz_protect_pet_photo_status on pets.pets;

drop function moderation.sync_pet_media_status_to_post();
drop function moderation.protect_post_status();
drop function moderation.protect_post_media_status();
drop function moderation.protect_pet_media_status();
drop function moderation.protect_pet_photo_status();

alter table social.posts      alter column moderation_status set default 'approved';
alter table social.post_media alter column moderation_status set default 'approved';
alter table pets.pet_media    alter column moderation_status set default 'approved';
alter table pets.pets         alter column profile_photo_moderation_status set default 'approved';

commit;

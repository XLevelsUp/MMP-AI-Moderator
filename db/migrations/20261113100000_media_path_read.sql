-- The worker checks that a media link is the file of the row it was sent for (12 Nov 2026), so it
-- reads each row's own path. For a database that ran 20261104100000_moderation_schema before then,
-- such as staging. A new database already has these grants; running it again changes nothing.
-- Undo with rollback/20261113100000_media_path_read_down.sql.

begin;

grant select (storage_path) on pets.pet_media to moderation_service;
grant select (storage_path) on social.post_media to moderation_service;
grant select (profile_photo_path) on pets.pets to moderation_service;

commit;

-- Undoes 20261113100000_media_path_read. Also set media.check_row_path: false, or the worker
-- can't read the paths and every job fails until it is sent to a moderator.

begin;

revoke select (storage_path) on pets.pet_media from moderation_service;
revoke select (storage_path) on social.post_media from moderation_service;
revoke select (profile_photo_path) on pets.pets from moderation_service;

commit;

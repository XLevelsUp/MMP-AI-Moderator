-- Undoes migration 1. Run the go-live rollback first if migration 2 was applied.
-- WARNING: this permanently deletes moderation.decisions (the audit records and moderators' answers)
-- and every status on existing rows. Export them first if they are needed.

begin;

-- The other admin panel grants go with the schema and the columns dropped below.
revoke select on pets.pet_media from service_role;

drop owned by moderation_service;
drop role moderation_service;

drop schema moderation cascade;

alter table social.posts      drop column moderation_status;
alter table social.post_media drop column moderation_status, drop column moderation_rejection_code;
alter table pets.pet_media    drop column moderation_status, drop column moderation_rejection_code;
alter table pets.pets         drop column profile_photo_moderation_status, drop column profile_photo_moderation_rejection_code;

commit;

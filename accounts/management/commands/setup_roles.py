from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission


class Command(BaseCommand):
    help = "Creates the Admin and Approver groups with the correct permissions."

    def handle(self, *args, **options):
        admin_group, _ = Group.objects.get_or_create(name="Admin")
        approver_group, _ = Group.objects.get_or_create(name="Approver")

        add_upload_perm = Permission.objects.get(codename="add_datasetupload")
        change_upload_perm = Permission.objects.get(codename="change_datasetupload")
        delete_upload_perm = Permission.objects.get(codename="delete_datasetupload")

        admin_group.permissions.set([add_upload_perm, delete_upload_perm])
        approver_group.permissions.set([change_upload_perm])

        self.stdout.write(
            self.style.SUCCESS("Admin and Approver groups created/updated.")
        )

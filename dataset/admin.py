from django.contrib import admin
from .models import EnrollmentRecord, DatasetUpload, PendingRecord


class DatasetUploadAdmin(admin.ModelAdmin):
    readonly_fields = (
        "approval_token",
        "verification_code",
        "uploaded_at",
        "reviewed_at",
    )
    list_display = (
        "id",
        "uploaded_by",
        "status",
        "total_rows",
        "clean_rows",
        "flagged_rows",
        "uploaded_at",
    )


admin.site.register(EnrollmentRecord)
admin.site.register(DatasetUpload, DatasetUploadAdmin)
admin.site.register(PendingRecord)

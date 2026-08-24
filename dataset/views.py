from django.contrib.auth.decorators import login_required, permission_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse
from django.utils import timezone
from django.db import transaction
from django.views.decorators.http import require_POST

from .forms import DatasetUploadForm
from .models import DatasetUpload, PendingRecord, EnrollmentRecord
from .validation import parse_and_validate_csv


@login_required
@permission_required("dataset.add_datasetupload", raise_exception=True)
def upload_csv(request):
    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = DatasetUpload.objects.create(
                uploaded_by=request.user,
                file=form.cleaned_data["file"],
                status="processing",
            )

            upload.file.seek(0)
            rows = parse_and_validate_csv(upload.file)

            clean_count = sum(1 for r in rows if r["status"] == "clean")
            flagged_count = len(rows) - clean_count

            for row in rows:
                PendingRecord.objects.create(
                    upload=upload,
                    validation_status=row["status"],
                    flag_reason=row["reason"],
                    **row["data"],
                )

            upload.total_rows = len(rows)
            upload.clean_rows = clean_count
            upload.flagged_rows = flagged_count
            upload.status = "pending_approval"
            upload.save()

            _send_approval_email(request, upload)

            messages.success(
                request,
                f"Uploaded {len(rows)} rows ({clean_count} clean, {flagged_count} flagged). "
                "An approval email has been sent.",
            )
            return redirect("dataset:upload_status", pk=upload.pk)
    else:
        form = DatasetUploadForm()

    return render(request, "dataset/upload_csv.html", {"form": form})


def _send_approval_email(request, upload):
    review_url = request.build_absolute_uri(
        reverse("dataset:review_upload", args=[upload.approval_token])
    )
    send_mail(
        subject="[Enrollment Predictor] New dataset upload needs approval",
        message=(
            f"{upload.uploaded_by} uploaded a new dataset ({upload.total_rows} rows: "
            f"{upload.clean_rows} clean, {upload.flagged_rows} flagged).\n\n"
            f"Review it here (login required): {review_url}\n\n"
            f"Verification code: {upload.verification_code}\n\n"
            f"You will need to enter this code on the review page before you can approve or reject.\n\n"
            f"This link expires in 48 hours."
        ),
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=settings.DATASET_APPROVER_EMAILS,
    )


@login_required
def upload_status(request, pk):
    upload = get_object_or_404(DatasetUpload, pk=pk)
    return render(request, "dataset/upload_status.html", {"upload": upload})


@login_required
@permission_required("dataset.change_datasetupload", raise_exception=True)
@login_required
@permission_required("dataset.change_datasetupload", raise_exception=True)
def review_upload(request, token):
    upload = get_object_or_404(DatasetUpload, approval_token=token)

    if not upload.token_is_valid():
        return render(request, "dataset/review_expired.html", {"upload": upload})

    verified_key = f"verified_upload_{upload.pk}"

    if not request.session.get(verified_key):
        error = None
        if request.method == "POST":
            entered_code = request.POST.get("verification_code", "").strip()
            if entered_code == upload.verification_code:
                request.session[verified_key] = True
                return redirect("dataset:review_upload", token=token)
            else:
                error = "Incorrect verification code. Check the approval email and try again."
        return render(
            request, "dataset/verify_code.html", {"upload": upload, "error": error}
        )

    clean_records = upload.records.filter(validation_status="clean", decision="pending")
    flagged_records = upload.records.filter(
        validation_status="flagged", decision="pending"
    )

    return render(
        request,
        "dataset/review_upload.html",
        {
            "upload": upload,
            "clean_records": clean_records,
            "flagged_records": flagged_records,
        },
    )

    return render(
        request,
        "dataset/review_upload.html",
        {
            "upload": upload,
            "clean_records": clean_records,
            "flagged_records": flagged_records,
        },
    )


@login_required
@permission_required("dataset.change_datasetupload", raise_exception=True)
@require_POST
def approve_upload(request, token):
    upload = get_object_or_404(DatasetUpload, approval_token=token)
    if not upload.token_is_valid():
        return render(request, "dataset/review_expired.html", {"upload": upload})

    upload.records.filter(validation_status="clean", decision="pending").update(
        decision="approved"
    )

    approved_flagged_ids = request.POST.getlist("approve_flagged")
    upload.records.filter(
        id__in=approved_flagged_ids, validation_status="flagged"
    ).update(decision="approved")
    upload.records.filter(validation_status="flagged", decision="pending").exclude(
        id__in=approved_flagged_ids
    ).update(decision="rejected")

    field_names = [
        f.name
        for f in EnrollmentRecord._meta.get_fields()
        if f.concrete and not f.auto_created
    ]
    with transaction.atomic():
        for record in upload.records.filter(decision="approved"):
            EnrollmentRecord.objects.create(
                source_upload=upload,
                **{f: getattr(record, f) for f in field_names if hasattr(record, f)},
            )

    upload.status = "approved"
    upload.reviewed_by = request.user
    upload.reviewed_at = timezone.now()
    upload.save()

    messages.success(
        request, f"Upload #{upload.pk} approved and promoted to the training dataset."
    )
    return redirect("dataset:upload_status", pk=upload.pk)


@login_required
@permission_required("dataset.change_datasetupload", raise_exception=True)
@require_POST
def reject_upload(request, token):
    upload = get_object_or_404(DatasetUpload, approval_token=token)
    upload.records.filter(decision="pending").update(decision="rejected")
    upload.status = "rejected"
    upload.reviewed_by = request.user
    upload.reviewed_at = timezone.now()
    upload.save()

    messages.info(request, f"Upload #{upload.pk} rejected. No records were added.")
    return redirect("dataset:upload_status", pk=upload.pk)


@login_required
@permission_required("dataset.change_datasetupload", raise_exception=True)
def pending_uploads(request):
    uploads = (
        DatasetUpload.objects.exclude(status="approved")
        .exclude(status="rejected")
        .order_by("-uploaded_at")
    )
    return render(request, "dataset/pending_uploads.html", {"uploads": uploads})

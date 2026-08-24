from django.urls import path
from . import views

app_name = "dataset"

urlpatterns = [
    path("upload/", views.upload_csv, name="upload_csv"),
    path("upload/<int:pk>/status/", views.upload_status, name="upload_status"),
    path("review/<uuid:token>/", views.review_upload, name="review_upload"),
    path("review/<uuid:token>/approve/", views.approve_upload, name="approve_upload"),
    path("review/<uuid:token>/reject/", views.reject_upload, name="reject_upload"),
    path("pending/", views.pending_uploads, name="pending_uploads"),
]

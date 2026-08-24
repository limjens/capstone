import uuid
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone

# --- Choice constants, matching your codebook + questionnaire wording ---

GENDER_CHOICES = [(1, "Male"), (2, "Female"), (3, "Prefer not to say")]

MUNICIPALITY_CHOICES = [
    (1, "San Juan"),
    (2, "St. Bernard"),
    (3, "Anahawan"),
    (4, "Hinunangan"),
    (5, "Hinundayan"),
    (6, "Silago"),
    (7, "Other"),
]

DISTANCE_CHOICES = [
    (1, "Less than 5 km"),
    (2, "5-10 km"),
    (3, "11-20 km"),
    (4, "More than 20 km"),
]

TRANSPORT_MODE_CHOICES = [
    (1, "Walking"),
    (2, "Motorcycle"),
    (3, "Public transport"),
    (4, "Private vehicle"),
]

TRANSPORT_COST_CHOICES = [
    (1, "P0-50"),
    (2, "P51-100"),
    (3, "P101-150"),
    (4, "P151 and above"),
]

INCOME_CHOICES = [
    (1, "Below P10,000"),
    (2, "P10,001-P20,000"),
    (3, "P20,001-P30,000"),
    (4, "Above P30,000"),
]

PARENT_EDUCATION_CHOICES = [
    (1, "Elementary"),
    (2, "High School"),
    (3, "College Level"),
    (4, "College Graduate"),
]

YES_NO_CHOICES = [(1, "Yes"), (0, "No")]

AFFECTS_CHOICE_CHOICES = [
    (1, "Not at all"),
    (2, "Slightly"),
    (3, "Moderately"),
    (4, "Strongly"),
]

SHS_TRACK_CHOICES = [
    (1, "STEM"),
    (2, "HUMSS"),
    (3, "ABM"),
    (4, "TVL"),
    (5, "GAS"),
]

GWA_BAND_CHOICES = [
    (4, "90 and above"),
    (3, "85-89"),
    (2, "80-84"),
    (1, "Below 80"),
]

SELF_PERFORMANCE_CHOICES = [
    (4, "Excellent"),
    (3, "Good"),
    (2, "Average"),
    (1, "Needs improvement"),
]

CAMPUS_CHOICES = [
    (1, "SLSU San Juan"),
    (2, "Other SLSU campuses"),
    (3, "Other universities/colleges"),
]

PROGRAM_GROUP_CHOICES = [
    (1, "BS Information Technology / Computer Studies"),
    (2, "Education / Teacher Education"),
    (3, "Agriculture / Fisheries / Science"),
    (4, "Entrepreneurship"),
    (5, "Industrial Technology / Electrical / Automotive"),
    (6, "Accountancy"),
    (7, "Office Administration"),
    (8, "Other"),
]

DEGREE_PROGRAM_CHOICES = [
    (1, "BIT"),
    (2, "BSED"),
    (3, "BTLED"),
    (4, "BSA"),
    (5, "BSE"),
    (6, "BSOA"),
    (7, "BSIT"),
    (8, "BSMA"),
    (9, "Undecided"),
]

PROGRAM_CHOICE_FACTOR_CHOICES = [
    (1, "Job opportunities"),
    (2, "Personal interest"),
    (3, "Availability of facilities"),
    (4, "Recommendation from family"),
]

DISCOURAGE_CHOICES = [
    (1, "Not at all"),
    (2, "Slightly"),
    (3, "Moderately"),
    (4, "Significantly"),
    (5, "Very strongly"),
]


class SurveyResponseFields(models.Model):
    """
    Abstract base holding every codebook field, shared by both the
    verified EnrollmentRecord and the unverified PendingRecord. No
    PII fields here (name/age/school) -- your questionnaire's consent
    form promises confidentiality and your analyzed dataset never
    stored those, so this model doesn't either.
    """

    gender = models.IntegerField(choices=GENDER_CHOICES, null=True, blank=True)
    municipality_residence_code = models.IntegerField(
        choices=MUNICIPALITY_CHOICES, null=True, blank=True
    )
    distance_to_campus = models.IntegerField(
        choices=DISTANCE_CHOICES, null=True, blank=True
    )
    transport_mode = models.IntegerField(
        choices=TRANSPORT_MODE_CHOICES, null=True, blank=True
    )
    transport_cost = models.IntegerField(
        choices=TRANSPORT_COST_CHOICES, null=True, blank=True
    )

    household_income = models.IntegerField(
        choices=INCOME_CHOICES, null=True, blank=True
    )
    parent_education = models.IntegerField(
        choices=PARENT_EDUCATION_CHOICES, null=True, blank=True
    )
    gov_assistance = models.IntegerField(choices=YES_NO_CHOICES, null=True, blank=True)
    financial_aid_affects_choice = models.IntegerField(
        choices=AFFECTS_CHOICE_CHOICES, null=True, blank=True
    )

    shs_track = models.IntegerField(choices=SHS_TRACK_CHOICES, null=True, blank=True)
    gwa_band = models.IntegerField(choices=GWA_BAND_CHOICES, null=True, blank=True)
    self_assessed_performance = models.IntegerField(
        choices=SELF_PERFORMANCE_CHOICES, null=True, blank=True
    )

    pursue_college = models.IntegerField(choices=YES_NO_CHOICES, null=True, blank=True)
    planned_campus = models.IntegerField(choices=CAMPUS_CHOICES, null=True, blank=True)

    program_group = models.IntegerField(
        choices=PROGRAM_GROUP_CHOICES, null=True, blank=True
    )
    intended_degree_program = models.IntegerField(
        choices=DEGREE_PROGRAM_CHOICES, null=True, blank=True
    )
    program_choice_factor = models.IntegerField(
        choices=PROGRAM_CHOICE_FACTOR_CHOICES, null=True, blank=True
    )
    labs_affect_decision = models.IntegerField(
        choices=AFFECTS_CHOICE_CHOICES, null=True, blank=True
    )

    enroll_if_course_unavailable = models.IntegerField(
        choices=YES_NO_CHOICES, null=True, blank=True
    )
    switch_if_slots_full = models.IntegerField(
        choices=YES_NO_CHOICES, null=True, blank=True
    )

    discourage_distance = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )
    discourage_tuition = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )
    discourage_limited_offerings = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )
    discourage_no_preferred_course = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )
    discourage_campus_reputation = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )
    discourage_facilities = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )
    discourage_scholarship_unavailable = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )
    discourage_family_feedback = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )
    discourage_relocation = models.IntegerField(
        choices=DISCOURAGE_CHOICES, null=True, blank=True
    )

    class Meta:
        abstract = True


class DatasetUpload(models.Model):
    STATUS_CHOICES = [
        ("processing", "Processing"),
        ("pending_approval", "Pending Approval"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
    ]

    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="dataset_uploads",
    )
    file = models.FileField(upload_to="dataset_uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="processing"
    )

    total_rows = models.IntegerField(default=0)
    clean_rows = models.IntegerField(default=0)
    flagged_rows = models.IntegerField(default=0)

    approval_token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    verification_code = models.CharField(max_length=6, blank=True)
    token_expires_at = models.DateTimeField(null=True, blank=True)

    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="dataset_reviews",
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.token_expires_at:
            self.token_expires_at = timezone.now() + timedelta(hours=48)
        if not self.verification_code:
            import random

            self.verification_code = f"{random.randint(0, 999999):06d}"
        super().save(*args, **kwargs)

    def token_is_valid(self):
        return (
            self.status == "pending_approval" and timezone.now() < self.token_expires_at
        )

    def __str__(self):
        return f"Upload #{self.pk} by {self.uploaded_by} ({self.status})"


class EnrollmentRecord(SurveyResponseFields):
    """The VERIFIED, model-training dataset. Only rows that made it
    through upload -> validation -> email approval land here."""

    added_at = models.DateTimeField(auto_now_add=True)
    source_upload = models.ForeignKey(
        DatasetUpload,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="promoted_records",
    )

    def __str__(self):
        return f"EnrollmentRecord #{self.pk}"


class PendingRecord(SurveyResponseFields):
    """Rows from a not-yet-approved CSV upload. Never used for
    training -- only promoted to EnrollmentRecord after approval."""

    VALIDATION_CHOICES = [("clean", "Clean"), ("flagged", "Flagged")]

    upload = models.ForeignKey(
        DatasetUpload, on_delete=models.CASCADE, related_name="records"
    )
    validation_status = models.CharField(
        max_length=10, choices=VALIDATION_CHOICES, default="clean"
    )
    flag_reason = models.TextField(blank=True)
    decision = models.CharField(
        max_length=10,
        choices=[
            ("pending", "Pending"),
            ("approved", "Approved"),
            ("rejected", "Rejected"),
        ],
        default="pending",
    )

    def __str__(self):
        return f"PendingRecord #{self.pk} ({self.validation_status})"

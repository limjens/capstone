from django.db import models


class PredictionResult(models.Model):
    calculated_at = models.DateTimeField(auto_now=True)

    # ============================================================
    # GENERAL
    # ============================================================

    total_students = models.IntegerField(default=0)

    # ============================================================
    # HEADCOUNT
    # ============================================================

    expected_headcount = models.FloatField(default=0)
    actual_headcount = models.IntegerField(default=0)

    pursue_rate = models.FloatField(default=0)

    headcount_accuracy = models.FloatField(default=0)
    headcount_min_accuracy = models.FloatField(default=0)
    headcount_max_accuracy = models.FloatField(default=0)

    # ============================================================
    # DEGREE PROGRAMS
    # ============================================================

    expected_undecided = models.FloatField(default=0)

    overall_program_accuracy = models.FloatField(default=0)

    top_program = models.CharField(max_length=100, default="N/A")

    # ============================================================
    # STUDENT SUPPORT
    # ============================================================

    top_barrier = models.CharField(max_length=200, default="N/A")

    overall_barrier_avg = models.FloatField(default=0)

    # ============================================================
    # DETAILED RESULTS
    # ============================================================

    program_results = models.JSONField(default=list)

    support_results = models.JSONField(default=list)

    model_results = models.JSONField(default=dict)

    # ============================================================
    # DISPLAY
    # ============================================================

    def __str__(self):
        return f"Prediction Result - {self.calculated_at}"

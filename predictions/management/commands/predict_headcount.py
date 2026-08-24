import random
import pandas as pd
import joblib
from django.core.management.base import BaseCommand
from django.conf import settings
from sklearn.model_selection import train_test_split

from dataset.models import EnrollmentRecord

MODELS_DIR = settings.BASE_DIR / "predictions" / "ml_models"

DISCOURAGE_COLS = [
    "discourage_distance",
    "discourage_tuition",
    "discourage_limited_offerings",
    "discourage_no_preferred_course",
    "discourage_campus_reputation",
    "discourage_facilities",
    "discourage_scholarship_unavailable",
    "discourage_family_feedback",
    "discourage_relocation",
]


class Command(BaseCommand):
    help = "Computes the expected SLSU San Juan enrollment headcount, plus an honest accuracy rate via repeated holdout testing."

    def handle(self, *args, **options):
        model_a_bundle = joblib.load(MODELS_DIR / "model_a_pursue_college.joblib")
        model_b_bundle = joblib.load(MODELS_DIR / "model_b_planned_campus.joblib")
        model_a, features_a = model_a_bundle["model"], model_a_bundle["features"]
        model_b, features_b = model_b_bundle["model"], model_b_bundle["features"]

        qs = EnrollmentRecord.objects.all().values()
        df = pd.DataFrame.from_records(qs)

        for col in DISCOURAGE_COLS + ["distance_to_campus", "transport_cost"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["barrier_score_total"] = df[DISCOURAGE_COLS].sum(axis=1, skipna=False)
        df["barrier_score_avg"] = df[DISCOURAGE_COLS].mean(axis=1, skipna=False)
        df["access_cost_index"] = df["distance_to_campus"] + df["transport_cost"]

        batch = df.dropna(subset=features_a + ["pursue_college"])

        # --- Full-batch number (optimistic -- models trained on this data) ---
        class_idx_a = list(model_a.classes_).index(1.0)
        p_pursue = model_a.predict_proba(batch[features_a])[:, class_idx_a]
        class_idx_b = list(model_b.classes_).index(1.0)
        p_sanjuan = model_b.predict_proba(batch[features_b])[:, class_idx_b]
        expected_headcount = (p_pursue * p_sanjuan).sum()
        hard_count = (
            (batch["pursue_college"] == 1) & (batch["planned_campus"] == 1)
        ).sum()

        self.stdout.write(f"Batch size: {len(batch)}")
        self.stdout.write(
            self.style.SUCCESS(
                f"Expected SLSU San Juan headcount: {expected_headcount:.1f}"
            )
        )
        self.stdout.write(f"Hard-classification count (sanity check): {hard_count}")

        # --- Honest accuracy rate: repeated random holdouts ---
        self.stdout.write(
            "\nRunning 10 repeated holdout tests for an honest accuracy rate..."
        )
        accuracy_rates = []
        for seed in range(10):
            _, test_batch = train_test_split(
                batch,
                test_size=0.2,
                random_state=seed,
                stratify=batch["pursue_college"],
            )
            p_pursue_t = model_a.predict_proba(test_batch[features_a])[:, class_idx_a]
            p_sanjuan_t = model_b.predict_proba(test_batch[features_b])[:, class_idx_b]
            expected_t = (p_pursue_t * p_sanjuan_t).sum()
            actual_t = (
                (test_batch["pursue_college"] == 1)
                & (test_batch["planned_campus"] == 1)
            ).sum()

            if actual_t > 0:
                error_pct = abs(expected_t - actual_t) / actual_t * 100
                accuracy_rates.append(max(0.0, 100 - error_pct))

        mean_acc = sum(accuracy_rates) / len(accuracy_rates)
        self.stdout.write(
            self.style.SUCCESS(
                f"\nHeadcount accuracy rate: {mean_acc:.1f}% (range: {min(accuracy_rates):.1f}%-{max(accuracy_rates):.1f}%)"
            )
        )

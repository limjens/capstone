import pandas as pd
import joblib

from django.core.management.base import BaseCommand
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.base import clone


from dataset.models import EnrollmentRecord

MODELS_DIR = settings.BASE_DIR / "predictions" / "ml_models"


# ============================================================
# DISCOURAGEMENT FEATURES
# ============================================================

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


# ============================================================
# DJANGO MANAGEMENT COMMAND
# ============================================================


class Command(BaseCommand):

    help = (
        "Computes the expected SLSU San Juan enrollment "
        "headcount and evaluates headcount estimation "
        "accuracy using repeated honest holdout testing."
    )

    # ========================================================
    # HANDLE
    # ========================================================

    def handle(self, *args, **options):

        # ----------------------------------------------------
        # LOAD TRAINED MODELS
        # ----------------------------------------------------

        model_a_bundle = joblib.load(MODELS_DIR / "model_a_pursue_college.joblib")

        model_b_bundle = joblib.load(MODELS_DIR / "model_b_planned_campus.joblib")

        model_a = model_a_bundle["model"]
        features_a = model_a_bundle["features"]

        model_b = model_b_bundle["model"]
        features_b = model_b_bundle["features"]

        # ----------------------------------------------------
        # LOAD APPROVED RECORDS
        # ----------------------------------------------------

        qs = EnrollmentRecord.objects.all().values()

        df = pd.DataFrame.from_records(qs)

        if df.empty:

            self.stdout.write(self.style.ERROR("No EnrollmentRecord rows found."))

            return

        self.stdout.write(f"Total EnrollmentRecord rows: " f"{len(df)}")

        # ----------------------------------------------------
        # CONVERT NUMERIC COLUMNS
        # ----------------------------------------------------

        for col in DISCOURAGE_COLS + [
            "distance_to_campus",
            "transport_cost",
        ]:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        df["barrier_score_total"] = df[DISCOURAGE_COLS].sum(
            axis=1,
            skipna=False,
        )

        df["barrier_score_avg"] = df[DISCOURAGE_COLS].mean(
            axis=1,
            skipna=False,
        )

        df["access_cost_index"] = df["distance_to_campus"] + df["transport_cost"]

        # ====================================================
        # FULL-BATCH EXPECTED HEADCOUNT
        # ====================================================

        batch = df.dropna(subset=features_a).copy()

        if batch.empty:

            self.stdout.write(
                self.style.ERROR("No usable rows found for " "headcount prediction.")
            )

            return

        # ----------------------------------------------------
        # MODEL A PROBABILITY
        # ----------------------------------------------------

        class_idx_a = list(model_a.classes_).index(1)

        p_pursue = model_a.predict_proba(batch[features_a])[:, class_idx_a]

        # ----------------------------------------------------
        # MODEL B PROBABILITY
        # ----------------------------------------------------

        # Only use rows that have all Model B features.
        batch_b = batch.dropna(subset=features_b).copy()

        class_idx_b = list(model_b.classes_).index(1)

        p_sanjuan = model_b.predict_proba(batch_b[features_b])[:, class_idx_b]

        # ----------------------------------------------------
        # EXPECTED HEADCOUNT
        # ----------------------------------------------------

        expected_headcount = (
            p_pursue[batch.index.isin(batch_b.index)] * p_sanjuan
        ).sum()

        # ----------------------------------------------------
        # ACTUAL OBSERVED HEADCOUNT
        # ----------------------------------------------------

        hard_count = (
            (batch["pursue_college"] == 1) & (batch["planned_campus"] == 1)
        ).sum()

        # ----------------------------------------------------
        # DISPLAY FULL-BATCH RESULTS
        # ----------------------------------------------------

        self.stdout.write(f"\nBatch size: {len(batch)}")

        self.stdout.write(
            self.style.SUCCESS(
                "Expected SLSU San Juan headcount: " f"{expected_headcount:.1f}"
            )
        )

        self.stdout.write(f"Hard-classification count " f"(sanity check): {hard_count}")

        # ====================================================
        # HONEST REPEATED HOLDOUT VALIDATION
        # ====================================================

        self.stdout.write(
            self.style.NOTICE(
                "\nRunning 10 repeated holdout " "tests with model retraining..."
            )
        )

        # ----------------------------------------------------
        # VALIDATION DATASET
        #
        # Both Model A and Model B need their features
        # and targets available.
        # ----------------------------------------------------

        validation_batch = df.dropna(
            subset=(
                features_a
                + features_b
                + [
                    "pursue_college",
                    "planned_campus",
                ]
            )
        ).copy()

        if len(validation_batch) < 20:

            self.stdout.write(
                self.style.WARNING(
                    "Not enough complete rows for " "repeated holdout validation."
                )
            )

            return

        # ----------------------------------------------------
        # CHECK MODEL A TARGET
        # ----------------------------------------------------

        if validation_batch["pursue_college"].nunique() < 2:

            self.stdout.write(
                self.style.WARNING("pursue_college contains fewer " "than two classes.")
            )

            return

        accuracy_rates = []

        expected_values = []
        actual_values = []

        # ====================================================
        # 10 REPEATED HOLDOUTS
        # ====================================================

        for seed in range(10):

            # ------------------------------------------------
            # 1. SPLIT DATA
            # ------------------------------------------------

            train_batch, test_batch = train_test_split(
                validation_batch,
                test_size=0.20,
                random_state=seed,
                stratify=validation_batch["pursue_college"],
            )

            # ------------------------------------------------
            # 2. CREATE FRESH MODELS
            # ------------------------------------------------

            test_model_a = clone(model_a)

            test_model_b = clone(model_b)

            # ------------------------------------------------
            # 3. TRAIN MODEL A
            #
            # Only training data is used.
            # ------------------------------------------------

            test_model_a.fit(
                train_batch[features_a],
                train_batch["pursue_college"],
            )

            # ------------------------------------------------
            # 4. TRAIN MODEL B
            #
            # IMPORTANT:
            #
            # Model B is trained ONLY on students who
            # actually pursue college.
            #
            # This matches the actual cascade.
            # ------------------------------------------------

            train_pursuing = train_batch[train_batch["pursue_college"] == 1].copy()

            if (
                len(train_pursuing) < 2
                or train_pursuing["planned_campus"].nunique() < 2
            ):

                self.stdout.write(
                    self.style.WARNING(
                        f"Test {seed + 1}: "
                        "not enough training data "
                        "for Model B. Skipped."
                    )
                )

                continue

            test_model_b.fit(
                train_pursuing[features_b],
                train_pursuing["planned_campus"],
            )

            # ------------------------------------------------
            # 5. PREDICT MODEL A PROBABILITY
            # ------------------------------------------------

            class_idx_a_test = list(test_model_a.classes_).index(1)

            p_pursue_test = test_model_a.predict_proba(test_batch[features_a])[
                :, class_idx_a_test
            ]

            # ------------------------------------------------
            # 6. PREDICT MODEL B PROBABILITY
            # ------------------------------------------------

            class_idx_b_test = list(test_model_b.classes_).index(1)

            p_sanjuan_test = test_model_b.predict_proba(test_batch[features_b])[
                :, class_idx_b_test
            ]

            # ------------------------------------------------
            # 7. EXPECTED HEADCOUNT
            #
            # P(pursue college)
            #
            # multiplied by
            #
            # P(San Juan | pursue college)
            # ------------------------------------------------

            expected_test = (p_pursue_test * p_sanjuan_test).sum()

            # ------------------------------------------------
            # 8. ACTUAL HEADCOUNT
            # ------------------------------------------------

            actual_test = (
                (test_batch["pursue_college"] == 1)
                & (test_batch["planned_campus"] == 1)
            ).sum()

            expected_values.append(expected_test)

            actual_values.append(actual_test)

            # ------------------------------------------------
            # 9. HEADCOUNT ACCURACY
            # ------------------------------------------------

            if actual_test > 0:

                error_pct = abs(expected_test - actual_test) / actual_test * 100

                accuracy = max(
                    0.0,
                    100 - error_pct,
                )

                accuracy_rates.append(accuracy)

                self.stdout.write(
                    f"Test {seed + 1}: "
                    f"Expected = "
                    f"{expected_test:.1f}, "
                    f"Actual = "
                    f"{actual_test}, "
                    f"Headcount estimation "
                    f"accuracy = "
                    f"{accuracy:.1f}%"
                )

            else:

                self.stdout.write(
                    f"Test {seed + 1}: "
                    f"Expected = "
                    f"{expected_test:.1f}, "
                    f"Actual = 0, "
                    f"Accuracy = N/A"
                )

        # ====================================================
        # SUMMARY
        # ====================================================

        if accuracy_rates:

            mean_accuracy = sum(accuracy_rates) / len(accuracy_rates)

            self.stdout.write(
                self.style.SUCCESS(
                    "\nHeadcount estimation " "accuracy rate: " f"{mean_accuracy:.1f}%"
                )
            )

            self.stdout.write(
                f"Accuracy range: "
                f"{min(accuracy_rates):.1f}% "
                f"- "
                f"{max(accuracy_rates):.1f}%"
            )

            # ------------------------------------------------
            # AVERAGE EXPECTED HEADCOUNT
            # ------------------------------------------------

            average_expected = sum(expected_values) / len(expected_values)

            # ------------------------------------------------
            # AVERAGE ACTUAL HEADCOUNT
            # ------------------------------------------------

            average_actual = sum(actual_values) / len(actual_values)

            self.stdout.write(
                f"\nAverage expected headcount: " f"{average_expected:.1f}"
            )

            self.stdout.write(f"Average actual headcount: " f"{average_actual:.1f}")

        else:

            self.stdout.write(
                self.style.WARNING(
                    "\nUnable to calculate headcount "
                    "estimation accuracy because "
                    "all test sets had zero actual "
                    "enrollments."
                )
            )

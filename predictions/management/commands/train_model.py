from pyexpat import features

import pandas as pd
import joblib
from django.core.management.base import BaseCommand
from django.conf import settings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

BASE_FEATURES = [
    "gender",
    "municipality_residence_code",
    "distance_to_campus",
    "transport_mode",
    "transport_cost",
    "household_income",
    "parent_education",
    "gov_assistance",
    "financial_aid_affects_choice",
    "shs_track",
    "gwa_band",
    "self_assessed_performance",
    "program_choice_factor",
    "labs_affect_decision",
    "enroll_if_course_unavailable",
    "switch_if_slots_full",
    "barrier_score_total",
    "barrier_score_avg",
    "access_cost_index",
] + DISCOURAGE_COLS

CANDIDATE_MODELS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=42
    ),
}


class Command(BaseCommand):
    help = "Trains the cascading enrollment prediction models from EnrollmentRecord data in the database."

    def handle(self, *args, **options):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        qs = EnrollmentRecord.objects.all().values()
        df = pd.DataFrame.from_records(qs)

        if df.empty:
            self.stdout.write(
                self.style.ERROR(
                    "No EnrollmentRecord rows found. Approve a dataset upload first."
                )
            )
            return

        for col in DISCOURAGE_COLS + ["distance_to_campus", "transport_cost"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["barrier_score_total"] = df[DISCOURAGE_COLS].sum(axis=1, skipna=False)
        df["barrier_score_avg"] = df[DISCOURAGE_COLS].mean(axis=1, skipna=False)
        df["access_cost_index"] = df["distance_to_campus"] + df["transport_cost"]

        stage_a = df.dropna(subset=BASE_FEATURES + ["pursue_college"])
        self.stdout.write(
            f"Training on {len(stage_a)} usable rows (of {len(df)} total)."
        )

        self._train_and_evaluate(
            stage_a, "pursue_college", BASE_FEATURES, "model_a_pursue_college"
        )

        enrolling = stage_a[stage_a["pursue_college"] == 1]

        stage_b = enrolling.dropna(subset=["planned_campus"])
        self._train_and_evaluate(
            stage_b, "planned_campus", BASE_FEATURES, "model_b_planned_campus"
        )

        stage_c1 = enrolling.dropna(subset=["program_group"])
        self._train_and_evaluate(
            stage_c1, "program_group", BASE_FEATURES, "model_c1_program_group"
        )

        stage_c2 = enrolling.dropna(subset=["intended_degree_program"])
        self._train_and_evaluate(
            stage_c2,
            "intended_degree_program",
            BASE_FEATURES,
            "model_c2_intended_degree_program",
        )

        self.stdout.write(
            self.style.SUCCESS(
                "Training complete. Models saved to predictions/ml_models/"
            )
        )

    def _train_and_evaluate(self, df, target_col, features, model_name):
        if len(df) < 20:
            self.stdout.write(
                self.style.WARNING(
                    f"Skipping {model_name}: not enough rows ({len(df)})."
                )
            )
            return

        X = df[features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        best_name, best_clf, best_f1, best_acc = None, None, -1, None
        for algo_name, make_clf in CANDIDATE_MODELS.items():
            clf = make_clf()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            self.stdout.write(f"  {model_name} [{algo_name}] acc={acc:.3f} f1={f1:.3f}")
            if f1 > best_f1:
                best_f1, best_acc, best_name, best_clf = f1, acc, algo_name, clf

        joblib.dump(
            {
                "model": best_clf,
                "features": features,
                "algorithm": best_name,
                "accuracy": round(float(best_acc), 4),
                "f1_score": round(float(best_f1), 4),
            },
            MODELS_DIR / f"{model_name}.joblib",
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"{model_name}: best={best_name} (f1={best_f1:.3f}) saved."
            )
        )

        X = df[features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        best_name, best_clf, best_f1 = None, None, -1
        for algo_name, make_clf in CANDIDATE_MODELS.items():
            clf = make_clf()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            self.stdout.write(f"  {model_name} [{algo_name}] acc={acc:.3f} f1={f1:.3f}")
            if f1 > best_f1:
                best_f1, best_name, best_clf = f1, algo_name, clf

        joblib.dump(
            {"model": best_clf, "features": features, "algorithm": best_name},
            MODELS_DIR / f"{model_name}.joblib",
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"{model_name}: best={best_name} (f1={best_f1:.3f}) saved."
            )
        )

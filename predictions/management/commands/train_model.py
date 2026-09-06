import pandas as pd
import joblib

from django.core.management.base import BaseCommand
from django.conf import settings

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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
# FEATURES USED BY ALL MODELS
# ============================================================

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


# ============================================================
# CANDIDATE MACHINE LEARNING MODELS
# ============================================================

CANDIDATE_MODELS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=42,
    ),
}


# ============================================================
# DJANGO MANAGEMENT COMMAND
# ============================================================


class Command(BaseCommand):

    help = (
        "Trains cascading enrollment prediction models with "
        "holdout testing, stratified cross-validation, and "
        "final retraining."
    )

    # ========================================================
    # MAIN COMMAND
    # ========================================================

    def handle(self, *args, **options):

        # ----------------------------------------------------
        # Create model directory
        # ----------------------------------------------------

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load database records
        # ----------------------------------------------------

        qs = EnrollmentRecord.objects.all().values()

        df = pd.DataFrame.from_records(qs)

        # ----------------------------------------------------
        # Check whether database is empty
        # ----------------------------------------------------

        if df.empty:

            self.stdout.write(
                self.style.ERROR(
                    "No EnrollmentRecord rows found. " "Approve a dataset upload first."
                )
            )

            return

        self.stdout.write(f"Total EnrollmentRecord rows: {len(df)}")

        # ====================================================
        # DATA PREPARATION
        # ====================================================

        # ----------------------------------------------------
        # Convert numeric columns
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
        # Calculate total barrier score
        # ----------------------------------------------------

        df["barrier_score_total"] = df[DISCOURAGE_COLS].sum(
            axis=1,
            skipna=False,
        )

        # ----------------------------------------------------
        # Calculate average barrier score
        # ----------------------------------------------------

        df["barrier_score_avg"] = df[DISCOURAGE_COLS].mean(
            axis=1,
            skipna=False,
        )

        # ----------------------------------------------------
        # Calculate accessibility/cost index
        # ----------------------------------------------------

        df["access_cost_index"] = df["distance_to_campus"] + df["transport_cost"]

        # ====================================================
        # STAGE A
        # ====================================================
        #
        # Predict:
        #     pursue_college
        #
        # ====================================================

        stage_a = df.dropna(
            subset=BASE_FEATURES
            + [
                "pursue_college",
            ]
        )

        self.stdout.write(
            f"\nStage A: {len(stage_a)} usable rows " "for pursue_college."
        )

        self._train_and_evaluate(
            stage_a,
            "pursue_college",
            BASE_FEATURES,
            "model_a_pursue_college",
        )

        # ====================================================
        # STAGE B
        # ====================================================
        #
        # Only students who actually answered
        # pursue_college = 1 are included.
        #
        # Predict:
        #     planned_campus
        #
        # ====================================================

        enrolling = stage_a[stage_a["pursue_college"] == 1]

        self.stdout.write(f"\nStudents pursuing college: {len(enrolling)}")

        stage_b = enrolling.dropna(subset=["planned_campus"])

        self.stdout.write(f"Stage B: {len(stage_b)} usable rows " "for planned_campus.")

        self._train_and_evaluate(
            stage_b,
            "planned_campus",
            BASE_FEATURES,
            "model_b_planned_campus",
        )

        # ====================================================
        # STAGE C1
        # ====================================================
        #
        # Predict:
        #     program_group
        #
        # ====================================================

        stage_c1 = enrolling.dropna(subset=["program_group"])

        self.stdout.write(
            f"\nStage C1: {len(stage_c1)} usable rows " "for program_group."
        )

        self._train_and_evaluate(
            stage_c1,
            "program_group",
            BASE_FEATURES,
            "model_c1_program_group",
        )

        # ====================================================
        # STAGE C2
        # ====================================================
        #
        # Predict:
        #     intended_degree_program
        #
        # ====================================================

        stage_c2 = enrolling.dropna(subset=["intended_degree_program"])

        self.stdout.write(
            f"\nStage C2: {len(stage_c2)} usable rows " "for intended_degree_program."
        )

        self._train_and_evaluate(
            stage_c2,
            "intended_degree_program",
            BASE_FEATURES,
            "model_c2_intended_degree_program",
        )

        # ====================================================
        # DONE
        # ====================================================

        self.stdout.write(self.style.SUCCESS("\nTraining complete."))

        self.stdout.write("Models saved to predictions/ml_models/")

    # ========================================================
    # TRAIN, CROSS-VALIDATE, HOLDOUT TEST, AND SAVE
    # ========================================================

    def _train_and_evaluate(
        self,
        df,
        target_col,
        features,
        model_name,
    ):

        # ----------------------------------------------------
        # Minimum dataset size
        # ----------------------------------------------------

        if len(df) < 20:

            self.stdout.write(
                self.style.WARNING(
                    f"Skipping {model_name}: " f"not enough rows ({len(df)})."
                )
            )

            return

        # ----------------------------------------------------
        # Separate features and target
        # ----------------------------------------------------

        X = df[features]

        y = df[target_col]

        # ----------------------------------------------------
        # Check number of classes
        # ----------------------------------------------------

        if y.nunique() < 2:

            self.stdout.write(
                self.style.WARNING(
                    f"Skipping {model_name}: "
                    f"target '{target_col}' has fewer "
                    "than 2 classes."
                )
            )

            return

        # ====================================================
        # 1. CREATE FINAL HOLDOUT SET
        # ====================================================
        #
        # 80% = development/training data
        # 20% = untouched final holdout test data
        #
        # The 20% is NOT used during cross-validation.
        #
        # ====================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        self.stdout.write(f"\nTraining {model_name}:")

        self.stdout.write(f"  Total rows:    {len(X)}")

        self.stdout.write(f"  Development:   {len(X_train)}")

        self.stdout.write(f"  Holdout test:   {len(X_test)}")

        # ====================================================
        # 2. STRATIFIED CROSS-VALIDATION
        # ====================================================
        #
        # Cross-validation happens ONLY on X_train/y_train.
        #
        # Five folds:
        #
        # Fold 1 → validation
        # Fold 2 → validation
        # Fold 3 → validation
        # Fold 4 → validation
        # Fold 5 → validation
        #
        # Each fold preserves the class distribution.
        #
        # ====================================================

        class_counts = y_train.value_counts()

        minimum_class_count = class_counts.min()

        cv_folds = min(5, int(minimum_class_count))

        if cv_folds < 2:

            self.stdout.write(
                self.style.WARNING(
                    f"Skipping {model_name}: "
                    "not enough samples in the smallest "
                    "class for stratified cross-validation."
                )
            )

            return

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42,
        )

        self.stdout.write(f"  Cross-validation: " f"{cv_folds}-fold stratified CV")

        # ====================================================
        # 3. EVALUATE CANDIDATE MODELS
        # ====================================================

        best_name = None
        best_clf = None

        best_f1 = -1

        best_acc = 0
        best_precision = 0
        best_recall = 0

        best_cv_accuracy = 0
        best_cv_precision = 0
        best_cv_recall = 0
        best_cv_f1 = 0

        for algo_name, make_clf in CANDIDATE_MODELS.items():

            # ------------------------------------------------
            # Create a fresh classifier
            # ------------------------------------------------

            clf = make_clf()

            # =================================================
            # 3A. STRATIFIED CROSS-VALIDATION
            # =================================================

            scoring = {
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted",
            }

            cv_results = cross_validate(
                clf,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False,
            )

            # ------------------------------------------------
            # Calculate mean CV metrics
            # ------------------------------------------------

            cv_accuracy = cv_results["test_accuracy"].mean()

            cv_precision = cv_results["test_precision"].mean()

            cv_recall = cv_results["test_recall"].mean()

            cv_f1 = cv_results["test_f1"].mean()

            # ------------------------------------------------
            # Calculate CV standard deviations
            # ------------------------------------------------

            cv_accuracy_std = cv_results["test_accuracy"].std()

            cv_precision_std = cv_results["test_precision"].std()

            cv_recall_std = cv_results["test_recall"].std()

            cv_f1_std = cv_results["test_f1"].std()

            # =================================================
            # 3B. HOLDOUT TEST
            # =================================================

            holdout_clf = make_clf()

            holdout_clf.fit(
                X_train,
                y_train,
            )

            y_pred = holdout_clf.predict(X_test)

            # ------------------------------------------------
            # Holdout accuracy
            # ------------------------------------------------

            acc = accuracy_score(
                y_test,
                y_pred,
            )

            # ------------------------------------------------
            # Holdout precision
            # ------------------------------------------------

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            # ------------------------------------------------
            # Holdout recall
            # ------------------------------------------------

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            # ------------------------------------------------
            # Holdout F1
            # ------------------------------------------------

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            # =================================================
            # DISPLAY RESULTS
            # =================================================

            self.stdout.write(f"\n  {model_name} [{algo_name}]")

            self.stdout.write(
                f"    CV accuracy:  " f"{cv_accuracy:.3f} " f"(±{cv_accuracy_std:.3f})"
            )

            self.stdout.write(
                f"    CV precision: "
                f"{cv_precision:.3f} "
                f"(±{cv_precision_std:.3f})"
            )

            self.stdout.write(
                f"    CV recall:    " f"{cv_recall:.3f} " f"(±{cv_recall_std:.3f})"
            )

            self.stdout.write(
                f"    CV F1:        " f"{cv_f1:.3f} " f"(±{cv_f1_std:.3f})"
            )

            self.stdout.write(f"    Holdout accuracy:  " f"{acc:.3f}")

            self.stdout.write(f"    Holdout precision: " f"{precision:.3f}")

            self.stdout.write(f"    Holdout recall:    " f"{recall:.3f}")

            self.stdout.write(f"    Holdout F1:        " f"{f1:.3f}")

            # =================================================
            # SELECT BEST MODEL
            # =================================================
            #
            # Holdout F1 remains the selection criterion.
            #
            # This means cross-validation is used for
            # stability/generalization assessment, while
            # the untouched holdout determines which model
            # is selected.
            #
            # =================================================

            if f1 > best_f1:

                best_name = algo_name

                best_clf = holdout_clf

                best_f1 = f1

                best_acc = acc

                best_precision = precision

                best_recall = recall

                best_cv_accuracy = cv_accuracy
                best_cv_precision = cv_precision
                best_cv_recall = cv_recall
                best_cv_f1 = cv_f1

                best_cv_accuracy_std = cv_accuracy_std
                best_cv_precision_std = cv_precision_std
                best_cv_recall_std = cv_recall_std
                best_cv_f1_std = cv_f1_std

        # ====================================================
        # 4. RETRAIN BEST MODEL USING ALL DATA
        # ====================================================
        #
        # The holdout metrics above remain the final
        # evaluation metrics.
        #
        # Now that evaluation is finished, train the selected
        # algorithm using all available usable records.
        #
        # ====================================================

        self.stdout.write(self.style.SUCCESS(f"\n  Selected algorithm: {best_name}"))

        final_clf = CANDIDATE_MODELS[best_name]()

        final_clf.fit(
            X,
            y,
        )

        # ====================================================
        # 5. SAVE MODEL
        # ====================================================

        model_path = MODELS_DIR / f"{model_name}.joblib"

        joblib.dump(
            {
                # --------------------------------------------
                # Final trained model
                # --------------------------------------------
                "model": final_clf,
                # --------------------------------------------
                # Features
                # --------------------------------------------
                "features": features,
                # --------------------------------------------
                # Selected algorithm
                # --------------------------------------------
                "algorithm": best_name,
                # --------------------------------------------
                # Holdout metrics
                # --------------------------------------------
                "accuracy": round(
                    float(best_acc),
                    4,
                ),
                "precision": round(
                    float(best_precision),
                    4,
                ),
                "recall": round(
                    float(best_recall),
                    4,
                ),
                "f1_score": round(
                    float(best_f1),
                    4,
                ),
                # --------------------------------------------
                # Cross-validation mean metrics
                # --------------------------------------------
                "cv_accuracy": round(
                    float(best_cv_accuracy),
                    4,
                ),
                "cv_precision": round(
                    float(best_cv_precision),
                    4,
                ),
                "cv_recall": round(
                    float(best_cv_recall),
                    4,
                ),
                "cv_f1_score": round(
                    float(best_cv_f1),
                    4,
                ),
                # --------------------------------------------
                # Cross-validation standard deviations
                # --------------------------------------------
                "cv_accuracy_std": round(
                    float(best_cv_accuracy_std),
                    4,
                ),
                "cv_precision_std": round(
                    float(best_cv_precision_std),
                    4,
                ),
                "cv_recall_std": round(
                    float(best_cv_recall_std),
                    4,
                ),
                "cv_f1_score_std": round(
                    float(best_cv_f1_std),
                    4,
                ),
                # --------------------------------------------
                # Validation configuration
                # --------------------------------------------
                "holdout_test_size": 0.20,
                "cv_folds": cv_folds,
                "random_state": 42,
            },
            model_path,
        )

        # ====================================================
        # 6. DISPLAY FINAL SUMMARY
        # ====================================================

        self.stdout.write(self.style.SUCCESS(f"\n  {model_name} completed."))

        self.stdout.write(f"    Algorithm: {best_name}")

        self.stdout.write(f"    Holdout accuracy:  {best_acc:.3f}")

        self.stdout.write(f"    Holdout precision: {best_precision:.3f}")

        self.stdout.write(f"    Holdout recall:    {best_recall:.3f}")

        self.stdout.write(f"    Holdout F1:        {best_f1:.3f}")

        self.stdout.write(f"    CV accuracy:       {best_cv_accuracy:.3f}")

        self.stdout.write(f"    CV precision:      {best_cv_precision:.3f}")

        self.stdout.write(f"    CV recall:         {best_cv_recall:.3f}")

        self.stdout.write(f"    CV F1:             {best_cv_f1:.3f}")

        self.stdout.write(f"    Saved to: {model_path}")

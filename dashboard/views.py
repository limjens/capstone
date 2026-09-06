import pandas as pd
import joblib

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.cache import never_cache

from sklearn.base import clone
import json
from sklearn.model_selection import train_test_split
from django.http import StreamingHttpResponse

from dataset.models import EnrollmentRecord
from .models import PredictionResult

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
# DISCOURAGEMENT LABELS
# ============================================================

DISCOURAGE_LABELS = {
    "discourage_distance": "Distance from campus",
    "discourage_tuition": "Tuition cost",
    "discourage_limited_offerings": "Limited program offerings",
    "discourage_no_preferred_course": "No preferred course available",
    "discourage_campus_reputation": "Campus reputation",
    "discourage_facilities": "Facilities",
    "discourage_scholarship_unavailable": "Scholarship unavailability",
    "discourage_family_feedback": "Family feedback",
    "discourage_relocation": "Need to relocate",
}


# ============================================================
# DEGREE PROGRAM LABELS
# ============================================================

DEGREE_PROGRAM_LABELS = {
    1: "BIT",
    2: "BSED",
    3: "BTLED",
    4: "BSA",
    5: "BSE",
    6: "BSOA",
    7: "BSIT",
    8: "BSMA",
    9: "Undecided",
}


# ============================================================
# ACTUAL DEGREE PROGRAMS
# ============================================================

DEGREE_PROGRAM_CODES = set(range(1, 9))

UNDECIDED_CODE = 9


# ============================================================
# SECTION SIZE
# ============================================================

STUDENTS_PER_SECTION = 30


# ============================================================
# LOAD SAVED MODEL
# ============================================================


def _load_model(name):

    bundle = joblib.load(MODELS_DIR / f"{name}.joblib")

    return (
        bundle["model"],
        bundle["features"],
        bundle.get("algorithm", "unknown"),
        bundle.get("accuracy"),
        bundle.get("f1_score"),
        bundle.get("cv_accuracy"),
        bundle.get("cv_f1_score"),
        bundle.get("cv_accuracy_std"),
        bundle.get("cv_f1_score_std"),
    )


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================


def _load_clean_batch():
    """
    Load the ENTIRE accumulated enrollment dataset.

    Every approved CSV adds records to EnrollmentRecord.
    Therefore, this function intentionally retrieves ALL
    EnrollmentRecord objects instead of only the latest upload.

    Example:

        Upload #1 -> 1,198 records
        Upload #2 ->   500 records
        Upload #3 ->   700 records

        Calculation uses:
        1,198 + 500 + 700 = 2,398 records
    """

    # IMPORTANT:
    # Do NOT filter by source_upload here.
    # We want every approved record accumulated in the database.
    qs = EnrollmentRecord.objects.all().values()

    df = pd.DataFrame.from_records(qs)

    if df.empty:
        return df

    # Convert numeric survey fields to numbers.
    numeric_columns = [
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
        "pursue_college",
        "planned_campus",
        "program_group",
        "intended_degree_program",
        "program_choice_factor",
        "labs_affect_decision",
        "enroll_if_course_unavailable",
        "switch_if_slots_full",
        *DISCOURAGE_COLS,
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------

    # Total discouragement/barrier score
    df["barrier_score_total"] = df[DISCOURAGE_COLS].sum(
        axis=1,
        skipna=False,
    )

    # Average discouragement/barrier score
    df["barrier_score_avg"] = df[DISCOURAGE_COLS].mean(
        axis=1,
        skipna=False,
    )

    # Combined accessibility/cost index
    df["access_cost_index"] = df["distance_to_campus"] + df["transport_cost"]

    return df


# ============================================================
# HEADCOUNT VALIDATION
# ============================================================


def _calculate_headcount_accuracy(
    df,
    model_a,
    features_a,
    model_b,
    features_b,
):

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

        return {
            "mean_accuracy": None,
            "min_accuracy": None,
            "max_accuracy": None,
        }

    if validation_batch["pursue_college"].nunique() < 2:

        return {
            "mean_accuracy": None,
            "min_accuracy": None,
            "max_accuracy": None,
        }

    accuracy_rates = []

    # --------------------------------------------------------
    # 10 repeated holdouts
    # --------------------------------------------------------

    for seed in range(10):

        try:

            train_batch, test_batch = train_test_split(
                validation_batch,
                test_size=0.20,
                random_state=seed,
                stratify=validation_batch["pursue_college"],
            )

        except ValueError:
            continue

        # ----------------------------------------------------
        # Fresh Model A
        # ----------------------------------------------------

        test_model_a = clone(model_a)

        test_model_a.fit(
            train_batch[features_a],
            train_batch["pursue_college"],
        )

        # ----------------------------------------------------
        # Model B training data
        # ----------------------------------------------------

        train_pursuing = train_batch[train_batch["pursue_college"] == 1].copy()

        if len(train_pursuing) < 2 or train_pursuing["planned_campus"].nunique() < 2:
            continue

        # ----------------------------------------------------
        # Fresh Model B
        # ----------------------------------------------------

        test_model_b = clone(model_b)

        test_model_b.fit(
            train_pursuing[features_b],
            train_pursuing["planned_campus"],
        )

        # ----------------------------------------------------
        # Class indexes
        # ----------------------------------------------------

        try:

            class_idx_a = list(test_model_a.classes_).index(1)

            class_idx_b = list(test_model_b.classes_).index(1)

        except ValueError:

            continue

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        p_pursue = test_model_a.predict_proba(test_batch[features_a])[:, class_idx_a]

        p_sanjuan = test_model_b.predict_proba(test_batch[features_b])[:, class_idx_b]

        # ----------------------------------------------------
        # Expected headcount
        # ----------------------------------------------------

        expected = (p_pursue * p_sanjuan).sum()

        # ----------------------------------------------------
        # Actual headcount
        # ----------------------------------------------------

        actual = (
            (test_batch["pursue_college"] == 1) & (test_batch["planned_campus"] == 1)
        ).sum()

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        if actual > 0:

            error_pct = abs(expected - actual) / actual * 100

            accuracy = max(
                0.0,
                100 - error_pct,
            )

            accuracy_rates.append(accuracy)

    # --------------------------------------------------------
    # No valid tests
    # --------------------------------------------------------

    if not accuracy_rates:

        return {
            "mean_accuracy": None,
            "min_accuracy": None,
            "max_accuracy": None,
        }

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    return {
        "mean_accuracy": round(
            sum(accuracy_rates) / len(accuracy_rates),
            1,
        ),
        "min_accuracy": round(
            min(accuracy_rates),
            1,
        ),
        "max_accuracy": round(
            max(accuracy_rates),
            1,
        ),
    }


# ============================================================
# DEGREE PROGRAM BREAKDOWN
# ============================================================


# ============================================================
# DEGREE PROGRAM BREAKDOWN
# ============================================================


def _calculate_program_breakdown(
    df,
    model_a,
    features_a,
    model_c2,
    features_c2,
):
    """
    Calculates the predicted degree-program distribution.

    IMPORTANT:
    - Uses the already-trained Model A and Model C2 for the
      main prediction.
    - Does NOT retrain Model A/C2 separately for every program.
    - Per-program accuracy is calculated by comparing the
      expected count with the actual count in the valid batch.
    - Overall program allocation accuracy still uses 10
      repeated holdout validation tests.
    """

    # ========================================================
    # PREPARE VALID BATCH
    # ========================================================

    batch = df.dropna(
        subset=(
            features_a
            + features_c2
            + [
                "pursue_college",
                "intended_degree_program",
            ]
        )
    ).copy()

    if batch.empty:
        return [], "N/A", None, 0.0

    # ========================================================
    # MODEL A — COLLEGE PURSUIT
    # ========================================================

    try:
        class_idx_a = list(model_a.classes_).index(1)
    except ValueError:
        return [], "N/A", None, 0.0

    p_pursue = model_a.predict_proba(batch[features_a])[:, class_idx_a]

    # ========================================================
    # MODEL C2 — DEGREE PROGRAM
    # ========================================================

    proba_c2 = model_c2.predict_proba(batch[features_c2])

    # ========================================================
    # EXPECTED UNDECIDED
    # ========================================================

    expected_undecided = 0.0

    try:
        undecided_index = list(model_c2.classes_).index(UNDECIDED_CODE)

        expected_undecided = float((p_pursue * proba_c2[:, undecided_index]).sum())

    except ValueError:
        expected_undecided = 0.0

    # ========================================================
    # DEGREE PROGRAM RESULTS
    # ========================================================

    program_rows = []

    for i, cls in enumerate(model_c2.classes_):

        cls = int(cls)

        # ----------------------------------------------------
        # Skip Undecided
        # ----------------------------------------------------

        if cls == UNDECIDED_CODE:
            continue

        # ----------------------------------------------------
        # Only valid degree-program codes 1-8
        # ----------------------------------------------------

        if cls not in DEGREE_PROGRAM_CODES:
            continue

        # ----------------------------------------------------
        # EXPECTED STUDENTS
        # ----------------------------------------------------

        expected = float((p_pursue * proba_c2[:, i]).sum())

        # ----------------------------------------------------
        # ACTUAL STUDENTS
        # ----------------------------------------------------

        actual = int(
            (
                (batch["pursue_college"] == 1)
                & (batch["intended_degree_program"] == cls)
            ).sum()
        )

        # ----------------------------------------------------
        # ESTIMATION ACCURACY
        #
        # This compares the expected student count
        # against the actual student count.
        # ----------------------------------------------------

        if actual > 0:

            error_pct = abs(expected - actual) / actual * 100

            accuracy = max(
                0.0,
                100 - error_pct,
            )

            accuracy = round(
                accuracy,
                1,
            )

        else:
            accuracy = None

        # ----------------------------------------------------
        # PROGRAM ROW
        # ----------------------------------------------------

        program_rows.append(
            {
                "program": DEGREE_PROGRAM_LABELS.get(
                    cls,
                    str(cls),
                ),
                "expected_students": round(
                    expected,
                    1,
                ),
                "sections_needed": round(
                    expected / STUDENTS_PER_SECTION,
                    1,
                ),
                "accuracy": accuracy,
            }
        )

    # ========================================================
    # SORT PROGRAMS BY EXPECTED STUDENTS
    # ========================================================

    program_rows.sort(key=lambda r: -r["expected_students"])

    top_program = program_rows[0]["program"] if program_rows else "N/A"

    # ========================================================
    # OVERALL PROGRAM ALLOCATION ACCURACY
    #
    # This is the expensive validation portion.
    #
    # We keep the 10 repeated holdouts here, but instead
    # of doing them once for EVERY program, we perform the
    # validation once per holdout and compare ALL programs
    # during that same validation.
    #
    # This reduces the amount of model training dramatically.
    # ========================================================

    overall_accuracies = []

    for seed in range(10):

        try:

            train_batch, test_batch = train_test_split(
                batch,
                test_size=0.20,
                random_state=seed,
                stratify=batch["pursue_college"],
            )

        except ValueError:
            continue

        # ----------------------------------------------------
        # Fresh Model A
        # ----------------------------------------------------

        test_model_a = clone(model_a)

        test_model_a.fit(
            train_batch[features_a],
            train_batch["pursue_college"],
        )

        # ----------------------------------------------------
        # Students actually pursuing college
        # ----------------------------------------------------

        train_pursuing = train_batch[train_batch["pursue_college"] == 1].copy()

        if (
            len(train_pursuing) < 2
            or train_pursuing["intended_degree_program"].nunique() < 2
        ):
            continue

        # ----------------------------------------------------
        # Fresh Model C2
        # ----------------------------------------------------

        test_model_c2 = clone(model_c2)

        test_model_c2.fit(
            train_pursuing[features_c2],
            train_pursuing["intended_degree_program"],
        )

        # ----------------------------------------------------
        # Model A class index
        # ----------------------------------------------------

        try:

            class_idx_a_test = list(test_model_a.classes_).index(1)

        except ValueError:
            continue

        # ----------------------------------------------------
        # Model A probabilities
        # ----------------------------------------------------

        p_pursue_test = test_model_a.predict_proba(test_batch[features_a])[
            :, class_idx_a_test
        ]

        # ----------------------------------------------------
        # Model C2 probabilities
        # ----------------------------------------------------

        proba_c2_test = test_model_c2.predict_proba(test_batch[features_c2])

        # ----------------------------------------------------
        # Compare ALL program classes
        # in this ONE holdout.
        # ----------------------------------------------------

        total_abs_error = 0.0
        total_actual = 0

        for i, cls in enumerate(test_model_c2.classes_):

            cls = int(cls)

            expected_test = float((p_pursue_test * proba_c2_test[:, i]).sum())

            actual_test = int(
                (
                    (test_batch["pursue_college"] == 1)
                    & (test_batch["intended_degree_program"] == cls)
                ).sum()
            )

            total_abs_error += abs(expected_test - actual_test)

            total_actual += actual_test

        # ----------------------------------------------------
        # Accuracy for this holdout
        # ----------------------------------------------------

        if total_actual > 0:

            error_pct = total_abs_error / total_actual * 100

            accuracy = max(
                0.0,
                100 - error_pct,
            )

            overall_accuracies.append(accuracy)

    # ========================================================
    # OVERALL PROGRAM ACCURACY
    # ========================================================

    if overall_accuracies:

        overall_program_accuracy = round(
            sum(overall_accuracies) / len(overall_accuracies),
            1,
        )

    else:

        overall_program_accuracy = None

    # ========================================================
    # RETURN
    # ========================================================

    return (
        program_rows,
        top_program,
        overall_program_accuracy,
        round(
            expected_undecided,
            1,
        ),
    )


# ============================================================
# SUPPORT PRIORITIES
# ============================================================


def _calculate_support_priorities(df):

    support_batch = df.dropna(subset=DISCOURAGE_COLS).copy()

    if support_batch.empty:

        return (
            [],
            "N/A",
            0,
        )

    avg_scores = support_batch[DISCOURAGE_COLS].mean().sort_values(ascending=False)

    support_rows = [
        {
            "barrier": DISCOURAGE_LABELS[col],
            "avg_severity": round(
                float(value),
                2,
            ),
        }
        for col, value in avg_scores.items()
    ]

    top_barrier = support_rows[0]["barrier"] if support_rows else "N/A"

    overall_barrier_avg = round(
        float(support_batch["barrier_score_avg"].mean()),
        2,
    )

    return (
        support_rows,
        top_barrier,
        overall_barrier_avg,
    )


# ============================================================
# CALCULATE PREDICTIONS
# ============================================================


# ============================================================
# CALCULATE PREDICTIONS
# ============================================================


@login_required
@never_cache
def calculate_predictions(request):

    # ========================================================
    # GET
    # ========================================================

    if request.method != "POST":

        latest_result = PredictionResult.objects.order_by("-calculated_at").first()

        return render(
            request,
            "dashboard/calculate.html",
            {
                "latest_result": latest_result,
            },
        )

    # ========================================================
    # STREAM CALCULATION
    # ========================================================

    def calculation_stream():

        try:

            # =================================================
            # PROGRESS MESSAGE FUNCTION
            # =================================================

            def send(
                stage,
                message,
                progress,
                status="running",
                result=None,
            ):

                data = {
                    "stage": stage,
                    "message": message,
                    "progress": progress,
                    "status": status,
                }

                if result is not None:
                    data["result"] = result

                return json.dumps(data) + "\n"

            # =================================================
            # START
            # =================================================

            yield send(
                "start",
                "Starting calculation...",
                0,
            )

            # =================================================
            # LOAD DATASET
            # =================================================

            yield send(
                "dataset",
                "Loading enrollment dataset...",
                5,
            )

            df = _load_clean_batch()

            if df.empty:

                yield send(
                    "error",
                    "No enrollment records were found.",
                    100,
                    status="error",
                )

                return

            yield send(
                "dataset",
                (f"Loaded {len(df)} " "accumulated enrollment records."),
                10,
            )

            # =================================================
            # MODEL A
            # =================================================

            yield send(
                "model_a",
                ("Loading Model A " "— College Pursuit..."),
                12,
            )

            (
                model_a,
                features_a,
                algo_a,
                acc_a,
                f1_a,
                cv_acc_a,
                cv_f1_a,
                cv_acc_std_a,
                cv_f1_std_a,
            ) = _load_model("model_a_pursue_college")

            yield send(
                "model_a",
                ("Model A loaded successfully " f"({algo_a})."),
                15,
            )

            # =================================================
            # MODEL B
            # =================================================

            yield send(
                "model_b",
                ("Loading Model B " "— Planned Campus..."),
                17,
            )

            (
                model_b,
                features_b,
                algo_b,
                acc_b,
                f1_b,
                cv_acc_b,
                cv_f1_b,
                cv_acc_std_b,
                cv_f1_std_b,
            ) = _load_model("model_b_planned_campus")

            yield send(
                "model_b",
                ("Model B loaded successfully " f"({algo_b})."),
                20,
            )

            # =================================================
            # MODEL C2
            # =================================================

            yield send(
                "model_c2",
                ("Loading Model C2 " "— Degree Program..."),
                22,
            )

            (
                model_c2,
                features_c2,
                algo_c2,
                acc_c2,
                f1_c2,
                cv_acc_c2,
                cv_f1_c2,
                cv_acc_std_c2,
                cv_f1_std_c2,
            ) = _load_model("model_c2_intended_degree_program")

            yield send(
                "model_c2",
                ("Model C2 loaded successfully " f"({algo_c2})."),
                25,
            )

            # =================================================
            # PREPARE DATA
            # =================================================

            yield send(
                "prepare",
                ("Preparing valid records " "for prediction..."),
                28,
            )

            batch = df.dropna(
                subset=(
                    features_a
                    + features_b
                    + [
                        "pursue_college",
                        "planned_campus",
                    ]
                )
            ).copy()

            if batch.empty:

                yield send(
                    "error",
                    ("No valid records are " "available for calculation."),
                    100,
                    status="error",
                )

                return

            yield send(
                "prepare",
                (f"{len(batch)} valid records " "will be used for calculation."),
                30,
            )

            # =================================================
            # MODEL A PREDICTION
            # =================================================

            yield send(
                "prediction_a",
                ("Running Model A to calculate " "college-pursuit probabilities..."),
                33,
            )

            try:

                class_idx_a = list(model_a.classes_).index(1)

            except ValueError:

                yield send(
                    "error",
                    ("Model A does not contain " "class 1."),
                    100,
                    status="error",
                )

                return

            p_pursue = model_a.predict_proba(batch[features_a])[:, class_idx_a]

            yield send(
                "prediction_a",
                "Model A prediction completed.",
                36,
            )

            # =================================================
            # MODEL B PREDICTION
            # =================================================

            yield send(
                "prediction_b",
                ("Running Model B to calculate " "SLSU San Juan probabilities..."),
                39,
            )

            try:

                class_idx_b = list(model_b.classes_).index(1)

            except ValueError:

                yield send(
                    "error",
                    ("Model B does not contain " "class 1."),
                    100,
                    status="error",
                )

                return

            p_sanjuan = model_b.predict_proba(batch[features_b])[:, class_idx_b]

            yield send(
                "prediction_b",
                "Model B prediction completed.",
                42,
            )

            # =================================================
            # EXPECTED HEADCOUNT
            # =================================================

            yield send(
                "headcount",
                ("Calculating expected " "SLSU San Juan headcount..."),
                44,
            )

            expected_headcount = round(
                float((p_pursue * p_sanjuan).sum()),
                1,
            )

            actual_headcount = int(
                ((batch["pursue_college"] == 1) & (batch["planned_campus"] == 1)).sum()
            )

            pursue_rate = round(
                100 * batch["pursue_college"].mean(),
                1,
            )

            yield send(
                "headcount",
                (f"Expected headcount: " f"{expected_headcount} students"),
                47,
            )

            # =================================================
            # HEADCOUNT VALIDATION
            # =================================================

            yield send(
                "headcount_validation",
                ("Running headcount validation " "using 10 repeated holdout tests..."),
                50,
            )

            headcount_results = _calculate_headcount_accuracy(
                df,
                model_a,
                features_a,
                model_b,
                features_b,
            )

            mean_accuracy = headcount_results["mean_accuracy"]

            min_accuracy = headcount_results["min_accuracy"]

            max_accuracy = headcount_results["max_accuracy"]

            yield send(
                "headcount_validation",
                (
                    "Headcount validation completed. "
                    f"Average accuracy: "
                    f"{mean_accuracy}%"
                ),
                58,
            )

            # =================================================
            # DEGREE PROGRAM BREAKDOWN
            # =================================================

            yield send(
                "programs",
                ("Calculating predicted degree " "program distribution..."),
                60,
            )

            (
                program_rows,
                top_program,
                overall_program_accuracy,
                expected_undecided,
            ) = _calculate_program_breakdown(
                df,
                model_a,
                features_a,
                model_c2,
                features_c2,
            )

            yield send(
                "programs",
                ("Degree program distribution " "calculated successfully."),
                78,
            )

            # =================================================
            # SUPPORT PRIORITIES
            # =================================================

            yield send(
                "support",
                ("Calculating student " "support priorities..."),
                80,
            )

            support_batch = df.dropna(subset=DISCOURAGE_COLS).copy()

            if not support_batch.empty:

                avg_scores = (
                    support_batch[DISCOURAGE_COLS].mean().sort_values(ascending=False)
                )

                support_rows = [
                    {
                        "barrier": (DISCOURAGE_LABELS[col]),
                        "avg_severity": round(
                            float(val),
                            2,
                        ),
                    }
                    for col, val in avg_scores.items()
                ]

                top_barrier = support_rows[0]["barrier"] if support_rows else "N/A"

                overall_barrier_avg = round(
                    float(support_batch["barrier_score_avg"].mean()),
                    2,
                )

            else:

                support_rows = []

                top_barrier = "N/A"

                overall_barrier_avg = 0

            yield send(
                "support",
                ("Student support priorities " "calculated."),
                84,
            )

            # =================================================
            # MODEL RESULTS
            # =================================================

            yield send(
                "results",
                ("Preparing final " "calculation results..."),
                87,
            )

            model_results = {
                "model_a": {
                    "algorithm": algo_a,
                    "accuracy": acc_a,
                    "f1_score": f1_a,
                    "cv_accuracy": cv_acc_a,
                    "cv_f1_score": cv_f1_a,
                    "cv_accuracy_std": cv_acc_std_a,
                    "cv_f1_score_std": cv_f1_std_a,
                },
                "model_b": {
                    "algorithm": algo_b,
                    "accuracy": acc_b,
                    "f1_score": f1_b,
                    "cv_accuracy": cv_acc_b,
                    "cv_f1_score": cv_f1_b,
                    "cv_accuracy_std": cv_acc_std_b,
                    "cv_f1_score_std": cv_f1_std_b,
                },
                "model_c2": {
                    "algorithm": algo_c2,
                    "accuracy": acc_c2,
                    "f1_score": f1_c2,
                    "cv_accuracy": cv_acc_c2,
                    "cv_f1_score": cv_f1_c2,
                    "cv_accuracy_std": cv_acc_std_c2,
                    "cv_f1_score_std": cv_f1_std_c2,
                },
            }

            # =================================================
            # SAVE TO DATABASE
            # =================================================

            yield send(
                "save",
                ("Saving calculation results " "to the database..."),
                90,
            )

            result = PredictionResult.objects.create(
                # ---------------------------------------------
                # GENERAL
                # ---------------------------------------------
                total_students=len(batch),
                # ---------------------------------------------
                # HEADCOUNT
                # ---------------------------------------------
                expected_headcount=(expected_headcount),
                actual_headcount=(actual_headcount),
                pursue_rate=(pursue_rate),
                headcount_accuracy=(mean_accuracy if mean_accuracy is not None else 0),
                headcount_min_accuracy=(
                    min_accuracy if min_accuracy is not None else 0
                ),
                headcount_max_accuracy=(
                    max_accuracy if max_accuracy is not None else 0
                ),
                # ---------------------------------------------
                # PROGRAMS
                # ---------------------------------------------
                expected_undecided=(expected_undecided),
                overall_program_accuracy=(
                    overall_program_accuracy
                    if overall_program_accuracy is not None
                    else 0
                ),
                top_program=(top_program),
                # ---------------------------------------------
                # SUPPORT
                # ---------------------------------------------
                top_barrier=(top_barrier),
                overall_barrier_avg=(overall_barrier_avg),
                # ---------------------------------------------
                # JSON DATA
                # ---------------------------------------------
                program_results=(program_rows),
                support_results=(support_rows),
                model_results=(model_results),
            )

            # =================================================
            # COMPLETE
            # =================================================

            yield send(
                "complete",
                ("Calculation completed " "successfully."),
                100,
                status="complete",
                result={
                    "id": result.id,
                    "total_students": (len(batch)),
                    "expected_headcount": (expected_headcount),
                    "actual_headcount": (actual_headcount),
                    "pursue_rate": (pursue_rate),
                    "headcount_accuracy": (mean_accuracy),
                    "program_accuracy": (overall_program_accuracy),
                    "expected_undecided": (expected_undecided),
                    "top_program": (top_program),
                    "top_barrier": (top_barrier),
                },
            )

        # =====================================================
        # ERROR
        # =====================================================

        except Exception as e:

            yield send(
                "error",
                ("Calculation failed: " f"{str(e)}"),
                100,
                status="error",
            )

    # ========================================================
    # STREAM RESPONSE
    # ========================================================

    response = StreamingHttpResponse(
        calculation_stream(),
        content_type="application/x-ndjson",
    )

    response["Cache-Control"] = "no-cache, no-store, must-revalidate"

    response["X-Accel-Buffering"] = "no"

    return response


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================


@login_required
@never_cache
def overview(request):

    # ========================================================
    # GET LATEST SAVED RESULT
    # ========================================================

    result = PredictionResult.objects.order_by("-calculated_at").first()

    # ========================================================
    # NO CALCULATION YET
    # ========================================================

    if result is None:

        return render(
            request,
            "dashboard/overview.html",
            {
                "batch_size": 0,
                "expected_headcount": 0,
                "hard_count": 0,
                "pursue_rate": 0,
                "mean_accuracy": None,
                "min_accuracy": None,
                "max_accuracy": None,
                "program_rows": [],
                "top_program": "N/A",
                "overall_program_accuracy": None,
                "expected_undecided": 0,
                "support_rows": [],
                "top_barrier": "N/A",
                "overall_barrier_avg": 0,
                "algo_a": "N/A",
                "acc_a": None,
                "f1_a": None,
                "cv_acc_a": None,
                "cv_f1_a": None,
                "cv_acc_std_a": None,
                "cv_f1_std_a": None,
                "algo_b": "N/A",
                "acc_b": None,
                "f1_b": None,
                "cv_acc_b": None,
                "cv_f1_b": None,
                "cv_acc_std_b": None,
                "cv_f1_std_b": None,
                "algo_c2": "N/A",
                "acc_c2": None,
                "f1_c2": None,
                "cv_acc_c2": None,
                "cv_f1_c2": None,
                "cv_acc_std_c2": None,
                "cv_f1_std_c2": None,
                "calculated_at": None,
            },
        )

    # ========================================================
    # MODEL RESULTS
    # ========================================================

    model_results = result.model_results or {}

    model_a = model_results.get("model_a", {})

    model_b = model_results.get("model_b", {})

    model_c2 = model_results.get("model_c2", {})

    # ========================================================
    # RENDER SAVED RESULTS
    # ========================================================

    return render(
        request,
        "dashboard/overview.html",
        {
            # ------------------------------------------------
            # Basic
            # ------------------------------------------------
            "batch_size": result.total_students,
            "expected_headcount": (result.expected_headcount),
            "hard_count": (result.actual_headcount),
            "pursue_rate": (result.pursue_rate),
            # ------------------------------------------------
            # Headcount validation
            # ------------------------------------------------
            "mean_accuracy": (result.headcount_accuracy),
            "min_accuracy": (result.headcount_min_accuracy),
            "max_accuracy": (result.headcount_max_accuracy),
            # ------------------------------------------------
            # Programs
            # ------------------------------------------------
            "program_rows": (result.program_results),
            "top_program": (result.top_program),
            "overall_program_accuracy": (result.overall_program_accuracy),
            "expected_undecided": (result.expected_undecided),
            # ------------------------------------------------
            # Support
            # ------------------------------------------------
            "support_rows": (result.support_results),
            "top_barrier": (result.top_barrier),
            "overall_barrier_avg": (result.overall_barrier_avg),
            # ------------------------------------------------
            # Model A
            # ------------------------------------------------
            "algo_a": model_a.get(
                "algorithm",
                "N/A",
            ),
            "acc_a": model_a.get("accuracy"),
            "f1_a": model_a.get("f1_score"),
            "cv_acc_a": model_a.get("cv_accuracy"),
            "cv_f1_a": model_a.get("cv_f1_score"),
            "cv_acc_std_a": model_a.get("cv_accuracy_std"),
            "cv_f1_std_a": model_a.get("cv_f1_score_std"),
            # ------------------------------------------------
            # Model B
            # ------------------------------------------------
            "algo_b": model_b.get(
                "algorithm",
                "N/A",
            ),
            "acc_b": model_b.get("accuracy"),
            "f1_b": model_b.get("f1_score"),
            "cv_acc_b": model_b.get("cv_accuracy"),
            "cv_f1_b": model_b.get("cv_f1_score"),
            "cv_acc_std_b": model_b.get("cv_accuracy_std"),
            "cv_f1_std_b": model_b.get("cv_f1_score_std"),
            # ------------------------------------------------
            # Model C2
            # ------------------------------------------------
            "algo_c2": model_c2.get(
                "algorithm",
                "N/A",
            ),
            "acc_c2": model_c2.get("accuracy"),
            "f1_c2": model_c2.get("f1_score"),
            "cv_acc_c2": model_c2.get("cv_accuracy"),
            "cv_f1_c2": model_c2.get("cv_f1_score"),
            "cv_acc_std_c2": model_c2.get("cv_accuracy_std"),
            "cv_f1_std_c2": model_c2.get("cv_f1_score_std"),
            # ------------------------------------------------
            # Calculation timestamp
            # ------------------------------------------------
            "calculated_at": (result.calculated_at),
        },
    )

import pandas as pd
import joblib
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
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

STUDENTS_PER_SECTION = 40


def _load_model(name):
    bundle = joblib.load(MODELS_DIR / f"{name}.joblib")
    return (
        bundle["model"],
        bundle["features"],
        bundle.get("algorithm", "unknown"),
        bundle.get("accuracy"),
        bundle.get("f1_score"),
    )


def _load_clean_batch():
    qs = EnrollmentRecord.objects.all().values()
    df = pd.DataFrame.from_records(qs)

    for col in DISCOURAGE_COLS + ["distance_to_campus", "transport_cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["barrier_score_total"] = df[DISCOURAGE_COLS].sum(axis=1, skipna=False)
    df["barrier_score_avg"] = df[DISCOURAGE_COLS].mean(axis=1, skipna=False)
    df["access_cost_index"] = df["distance_to_campus"] + df["transport_cost"]
    return df


@login_required
def overview(request):
    model_a, features_a, algo_a, acc_a, f1_a = _load_model("model_a_pursue_college")
    model_b, features_b, algo_b, acc_b, f1_b = _load_model("model_b_planned_campus")
    model_c2, features_c2, algo_c2, acc_c2, f1_c2 = _load_model(
        "model_c2_intended_degree_program"
    )

    df = _load_clean_batch()
    batch = df.dropna(subset=features_a + ["pursue_college"])

    class_idx_a = list(model_a.classes_).index(1.0)
    p_pursue = model_a.predict_proba(batch[features_a])[:, class_idx_a]
    class_idx_b = list(model_b.classes_).index(1.0)
    p_sanjuan = model_b.predict_proba(batch[features_b])[:, class_idx_b]

    expected_headcount = round(float((p_pursue * p_sanjuan).sum()), 1)
    hard_count = int(
        ((batch["pursue_college"] == 1) & (batch["planned_campus"] == 1)).sum()
    )
    pursue_rate = round(100 * batch["pursue_college"].mean(), 1)

    # --- Headcount accuracy: repeated holdout ---
    accuracy_rates = []
    for seed in range(10):
        _, test_batch = train_test_split(
            batch, test_size=0.2, random_state=seed, stratify=batch["pursue_college"]
        )
        p_pursue_t = model_a.predict_proba(test_batch[features_a])[:, class_idx_a]
        p_sanjuan_t = model_b.predict_proba(test_batch[features_b])[:, class_idx_b]
        expected_t = (p_pursue_t * p_sanjuan_t).sum()
        actual_t = (
            (test_batch["pursue_college"] == 1) & (test_batch["planned_campus"] == 1)
        ).sum()
        if actual_t > 0:
            error_pct = abs(expected_t - actual_t) / actual_t * 100
            accuracy_rates.append(max(0.0, 100 - error_pct))
    mean_accuracy = round(sum(accuracy_rates) / len(accuracy_rates), 1)
    min_accuracy = round(min(accuracy_rates), 1)
    max_accuracy = round(max(accuracy_rates), 1)

    # --- Degree program breakdown WITH per-program accuracy ---
    proba_c2 = model_c2.predict_proba(batch[features_c2])
    program_rows = []
    for i, cls in enumerate(model_c2.classes_):
        expected = float((p_pursue * proba_c2[:, i]).sum())

        # per-program repeated holdout accuracy
        prog_accuracies = []
        for seed in range(10):
            _, test_batch = train_test_split(
                batch,
                test_size=0.2,
                random_state=seed,
                stratify=batch["pursue_college"],
            )
            p_pursue_t = model_a.predict_proba(test_batch[features_a])[:, class_idx_a]
            proba_c2_t = model_c2.predict_proba(test_batch[features_c2])[:, i]
            expected_t = (p_pursue_t * proba_c2_t).sum()
            actual_t = (
                (test_batch["pursue_college"] == 1)
                & (test_batch["intended_degree_program"] == cls)
            ).sum()
            if actual_t > 0:
                error_pct = abs(expected_t - actual_t) / actual_t * 100
                prog_accuracies.append(max(0.0, 100 - error_pct))
        prog_accuracy = (
            round(sum(prog_accuracies) / len(prog_accuracies), 1)
            if prog_accuracies
            else None
        )

        program_rows.append(
            {
                "program": DEGREE_PROGRAM_LABELS.get(int(cls), str(cls)),
                "expected_students": round(expected, 1),
                "sections_needed": round(expected / STUDENTS_PER_SECTION, 1),
                "accuracy": prog_accuracy,
            }
        )
    program_rows.sort(key=lambda r: -r["expected_students"])
    top_program = program_rows[0]["program"] if program_rows else "N/A"

    # --- Student support priorities ---
    avg_scores = batch[DISCOURAGE_COLS].mean().sort_values(ascending=False)
    support_rows = [
        {"barrier": DISCOURAGE_LABELS[col], "avg_severity": round(float(val), 2)}
        for col, val in avg_scores.items()
    ]
    top_barrier = support_rows[0]["barrier"] if support_rows else "N/A"
    overall_barrier_avg = round(float(batch["barrier_score_avg"].mean()), 2)

    return render(
        request,
        "dashboard/overview.html",
        {
            "batch_size": len(batch),
            "expected_headcount": expected_headcount,
            "hard_count": hard_count,
            "pursue_rate": pursue_rate,
            "mean_accuracy": mean_accuracy,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy,
            "program_rows": program_rows,
            "top_program": top_program,
            "support_rows": support_rows,
            "top_barrier": top_barrier,
            "overall_barrier_avg": overall_barrier_avg,
            "algo_a": algo_a,
            "acc_a": acc_a,
            "f1_a": f1_a,
            "algo_b": algo_b,
            "acc_b": acc_b,
            "f1_b": f1_b,
            "algo_c2": algo_c2,
            "acc_c2": acc_c2,
            "f1_c2": f1_c2,
        },
    )

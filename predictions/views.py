import joblib
from django.conf import settings
from django.shortcuts import render

from .forms import PredictionTestForm

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


def public_test(request):
    result = None

    if request.method == "POST":
        form = PredictionTestForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            data["barrier_score_total"] = sum(data[c] for c in DISCOURAGE_COLS)
            data["barrier_score_avg"] = data["barrier_score_total"] / len(
                DISCOURAGE_COLS
            )
            data["access_cost_index"] = (
                data["distance_to_campus"] + data["transport_cost"]
            )

            model_a_bundle = joblib.load(MODELS_DIR / "model_a_pursue_college.joblib")
            model_b_bundle = joblib.load(MODELS_DIR / "model_b_planned_campus.joblib")
            model_a, features_a = model_a_bundle["model"], model_a_bundle["features"]
            model_b, features_b = model_b_bundle["model"], model_b_bundle["features"]

            row_a = [[data[f] for f in features_a]]
            class_idx_a = list(model_a.classes_).index(1.0)
            p_pursue = model_a.predict_proba(row_a)[0][class_idx_a]

            row_b = [[data[f] for f in features_b]]
            class_idx_b = list(model_b.classes_).index(1.0)
            p_sanjuan = model_b.predict_proba(row_b)[0][class_idx_b]

            p_enroll_sanjuan = p_pursue * p_sanjuan

            result = {
                "p_pursue": round(p_pursue * 100, 1),
                "p_sanjuan_given_pursue": round(p_sanjuan * 100, 1),
                "p_enroll_sanjuan": round(p_enroll_sanjuan * 100, 1),
            }
    else:
        form = PredictionTestForm()

    return render(
        request, "predictions/public_test.html", {"form": form, "result": result}
    )

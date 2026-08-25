import csv
import io

# Same valid-code rules as ml/preprocessing.py — keep these in sync if
# the codebook ever changes.
VALID_CODES = {
    "gender": {1, 2, 3},
    "municipality_residence_code": {1, 2, 3, 4, 5, 6, 7},
    "distance_to_campus": {1, 2, 3, 4},
    "transport_mode": {1, 2, 3, 4},
    "transport_cost": {1, 2, 3, 4},
    "household_income": {1, 2, 3, 4},
    "parent_education": {1, 2, 3, 4},
    "gov_assistance": {0, 1},
    "financial_aid_affects_choice": {1, 2, 3, 4},
    "shs_track": {1, 2, 3, 4, 5},
    "gwa_band": {1, 2, 3, 4},
    "self_assessed_performance": {1, 2, 3, 4},
    "pursue_college": {0, 1},
    "planned_campus": {1, 2, 3},
    "program_group": {1, 2, 3, 4, 5, 6, 7, 8},
    "intended_degree_program": {1, 2, 3, 4, 5, 6, 7, 8, 9},
    "program_choice_factor": {1, 2, 3, 4},
    "labs_affect_decision": {1, 2, 3, 4},
    "enroll_if_course_unavailable": {0, 1},
    "switch_if_slots_full": {0, 1},
}

for col in [
    "discourage_distance",
    "discourage_tuition",
    "discourage_limited_offerings",
    "discourage_no_preferred_course",
    "discourage_campus_reputation",
    "discourage_facilities",
    "discourage_scholarship_unavailable",
    "discourage_family_feedback",
    "discourage_relocation",
]:
    VALID_CODES[col] = {1, 2, 3, 4, 5}

    REQUIRED_COLUMNS = set(VALID_CODES.keys())


def validate_csv_headers(uploaded_file):
    """
    Checks the CSV's header row against the expected survey columns
    BEFORE processing any rows. Returns (is_valid, missing_columns, extra_columns).
    """
    uploaded_file.seek(0)
    decoded = uploaded_file.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(decoded))
    actual_columns = set(reader.fieldnames or [])

    missing = REQUIRED_COLUMNS - actual_columns
    extra = actual_columns - REQUIRED_COLUMNS

    uploaded_file.seek(0)  # reset so it can be read again later
    return (len(missing) == 0, missing, extra)


def parse_and_validate_csv(uploaded_file):
    """
    Reads the uploaded CSV and returns a list of dicts, one per row:
        {"data": {...cleaned field values...}, "status": "clean"|"flagged", "reason": "..."}
    Rows aren't dropped here -- everything becomes a PendingRecord,
    just marked clean or flagged so the approver can see both.
    """
    decoded = uploaded_file.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(decoded))

    results = []
    for row in reader:
        cleaned = {}
        reasons = []

        for col, valid_set in VALID_CODES.items():
            raw_val = (row.get(col) or "").strip()
            if raw_val == "" or raw_val == "`":
                cleaned[col] = None
                continue
            try:
                num = int(float(raw_val))
            except ValueError:
                cleaned[col] = None
                reasons.append(f"{col}: non-numeric value '{raw_val}'")
                continue
            if num not in valid_set:
                cleaned[col] = None
                reasons.append(f"{col}: invalid code {num}")
            else:
                cleaned[col] = num

        status = "flagged" if reasons else "clean"
        results.append(
            {"data": cleaned, "status": status, "reason": "; ".join(reasons)}
        )

    return results

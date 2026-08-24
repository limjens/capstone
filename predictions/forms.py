from django import forms
from dataset.models import (
    GENDER_CHOICES,
    MUNICIPALITY_CHOICES,
    DISTANCE_CHOICES,
    TRANSPORT_MODE_CHOICES,
    TRANSPORT_COST_CHOICES,
    INCOME_CHOICES,
    PARENT_EDUCATION_CHOICES,
    YES_NO_CHOICES,
    AFFECTS_CHOICE_CHOICES,
    SHS_TRACK_CHOICES,
    GWA_BAND_CHOICES,
    SELF_PERFORMANCE_CHOICES,
    PROGRAM_CHOICE_FACTOR_CHOICES,
    DISCOURAGE_CHOICES,
)


class PredictionTestForm(forms.Form):
    gender = forms.TypedChoiceField(choices=GENDER_CHOICES, coerce=int, label="Gender")
    municipality_residence_code = forms.TypedChoiceField(
        choices=MUNICIPALITY_CHOICES, coerce=int, label="Municipality of residence"
    )
    distance_to_campus = forms.TypedChoiceField(
        choices=DISTANCE_CHOICES, coerce=int, label="Distance to campus"
    )
    transport_mode = forms.TypedChoiceField(
        choices=TRANSPORT_MODE_CHOICES, coerce=int, label="Usual transport mode"
    )
    transport_cost = forms.TypedChoiceField(
        choices=TRANSPORT_COST_CHOICES, coerce=int, label="Daily transport cost"
    )

    household_income = forms.TypedChoiceField(
        choices=INCOME_CHOICES, coerce=int, label="Household income"
    )
    parent_education = forms.TypedChoiceField(
        choices=PARENT_EDUCATION_CHOICES, coerce=int, label="Parent's highest education"
    )
    gov_assistance = forms.TypedChoiceField(
        choices=YES_NO_CHOICES, coerce=int, label="Receiving government assistance?"
    )
    financial_aid_affects_choice = forms.TypedChoiceField(
        choices=AFFECTS_CHOICE_CHOICES,
        coerce=int,
        label="Does financial aid affect your choice?",
    )

    shs_track = forms.TypedChoiceField(
        choices=SHS_TRACK_CHOICES, coerce=int, label="Senior High School track"
    )
    gwa_band = forms.TypedChoiceField(
        choices=GWA_BAND_CHOICES, coerce=int, label="General weighted average"
    )
    self_assessed_performance = forms.TypedChoiceField(
        choices=SELF_PERFORMANCE_CHOICES,
        coerce=int,
        label="Self-assessed academic performance",
    )

    program_choice_factor = forms.TypedChoiceField(
        choices=PROGRAM_CHOICE_FACTOR_CHOICES,
        coerce=int,
        label="Biggest factor in choosing a program",
    )
    labs_affect_decision = forms.TypedChoiceField(
        choices=AFFECTS_CHOICE_CHOICES,
        coerce=int,
        label="Do labs/facilities affect your decision?",
    )
    enroll_if_course_unavailable = forms.TypedChoiceField(
        choices=YES_NO_CHOICES,
        coerce=int,
        label="Would you still enroll if your preferred course is unavailable?",
    )
    switch_if_slots_full = forms.TypedChoiceField(
        choices=YES_NO_CHOICES,
        coerce=int,
        label="Would you switch programs if slots are full?",
    )

    discourage_distance = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much does distance discourage you?",
    )
    discourage_tuition = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much does tuition discourage you?",
    )
    discourage_limited_offerings = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much do limited program offerings discourage you?",
    )
    discourage_no_preferred_course = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much does having no preferred course discourage you?",
    )
    discourage_campus_reputation = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much does campus reputation discourage you?",
    )
    discourage_facilities = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much do facilities discourage you?",
    )
    discourage_scholarship_unavailable = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much does scholarship unavailability discourage you?",
    )
    discourage_family_feedback = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much does family feedback discourage you?",
    )
    discourage_relocation = forms.TypedChoiceField(
        choices=DISCOURAGE_CHOICES,
        coerce=int,
        label="How much does needing to relocate discourage you?",
    )

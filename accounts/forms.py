from django import forms
from django.contrib.auth.models import User, Group


class CreateAccountForm(forms.Form):
    username = forms.CharField(max_length=150)
    email = forms.EmailField(required=False)
    password1 = forms.CharField(widget=forms.PasswordInput, label="Password")
    password2 = forms.CharField(widget=forms.PasswordInput, label="Confirm password")
    role = forms.ChoiceField(
        choices=[
            ("Admin", "Admin (can upload datasets)"),
            ("Approver", "Approver (can review/approve)"),
            ("both", "Both"),
        ]
    )

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("password1") != cleaned.get("password2"):
            raise forms.ValidationError("Passwords do not match.")
        if User.objects.filter(username=cleaned.get("username")).exists():
            raise forms.ValidationError("That username is already taken.")
        return cleaned

from django import forms


class DatasetUploadForm(forms.Form):
    file = forms.FileField(
        label="CSV file",
        help_text="Must match the survey codebook column names.",
    )

    def clean_file(self):
        file = self.cleaned_data["file"]
        if not file.name.endswith(".csv"):
            raise forms.ValidationError("Please upload a .csv file.")
        return file

from django import forms

class RecruitmentForm(forms.Form):
    recruitment_info = forms.FileField()
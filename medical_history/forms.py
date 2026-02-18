from django import forms
from .models import MedicalHistory

class MedicalHistoryForm(forms.ModelForm):
    class Meta:
        model = MedicalHistory
        fields = ['allergies', 'diseases']
        labels = {
            'allergies': 'Your Allergies',
            'diseases': 'Medical Conditions'
        }
        help_texts = {
            'allergies': 'Separate with commas (e.g., milk, peanuts, eggs)',
            'diseases': 'Separate with commas (e.g., diabetes, asthma)',
        }
        widgets = {
            'allergies': forms.Textarea(attrs={
                'placeholder': 'e.g., Peanuts, Shellfish, Dairy',
                'class': 'form-control placeholder-visible',
                'rows': 2
            }),
            'diseases': forms.Textarea(attrs={
                'placeholder': 'e.g., Diabetes, Hypertension',
                'class': 'form-control placeholder-visible',
                'rows': 2
            }),
        }
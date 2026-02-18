from django.db import models
from django.contrib.auth.models import User

class MedicalHistory(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    allergies = models.TextField(help_text="Enter comma-separated list of allergies")
    diseases = models.TextField(help_text="Enter comma-separated list of past diagnoses")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Medical History - {self.user.username}"
    

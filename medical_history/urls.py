from django.urls import path
from .views import medical_history_view, check_medical_history, edit_medical_info

urlpatterns = [
    path('medical-info/', medical_history_view, name='medical_history'),
    path('check-medical/', check_medical_history, name='check_medical'),
    path('edit/', edit_medical_info, name='edit-medical'),
]
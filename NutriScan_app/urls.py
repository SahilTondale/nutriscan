from django import urls
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('', views.home, name='home'),
    path('upload', views.upload, name='upload'),
    path('analyze', views.analyze_ingredients, name='analyze'),
    path('history',views.history, name='history'),
    path('analysis/<int:analysis_id>', views.analysis_detail, name='analysis_detail'),
]
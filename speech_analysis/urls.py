from django.urls import path
from . import views

urlpatterns = [
    path('analyze-audio/', views.analyze_audio, name='analyze_audio'),
]

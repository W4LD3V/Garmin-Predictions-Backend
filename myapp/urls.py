from django.urls import path
from . import views

urlpatterns = [
    path("train/", views.train_model, name="train_model"),
    path("predict/", views.predict_distances, name="predict_distances"),
]

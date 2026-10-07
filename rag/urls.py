from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("lab/<slug:slug>/", views.lab, name="lab"),
    path("upload/", views.upload, name="upload"),
    path("guardrails/", views.guardrails_view, name="guardrails"),
    path("mlops/", views.mlops_view, name="mlops"),
    path("guide/", views.guide, name="guide"),
    path("api/ask/", views.api_ask, name="api_ask"),
    path("healthz", views.healthz, name="healthz"),
]

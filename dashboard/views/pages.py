# dashboard/views/pages.py
from django.shortcuts import render
from ._shared import ALL_RUNS, NORMAL_RUNS, ANOMALY_RUNS, MODEL_CHOICES


def index(request):
    return render(request, "dashboard/index.html", {
        "all_runs":      ALL_RUNS,
        "normal_runs":   NORMAL_RUNS,
        "anomaly_runs":  ANOMALY_RUNS,
        "model_choices": MODEL_CHOICES,
    })
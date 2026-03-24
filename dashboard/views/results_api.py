# dashboard/views/results_api.py
"""Results table API: all models x all thresholds x all runs."""
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from ._shared import (
    DF, MODEL_CHOICES, ALL_RUNS, NORMAL_RUNS,
    TH_NAMES, _cfg_from_request,
    compute_metrics_from_error,
)
from .batch import _get_batch_job

# ── Abort flag ────────────────────────────────────────────────────────────────
# Set to True via POST /api/results-abort/ to cancel an in-progress computation
_RESULTS_ABORT = {"flag": False}


@require_GET
def api_results_table(request):
    job = _get_batch_job(request)
    if not job:
        return JsonResponse({"error": "no completed batch job"}, status=400)

    # Reset abort flag at start of each new computation
    _RESULTS_ABORT["flag"] = False

    cfg        = _cfg_from_request(request)
    table_rows = []
    has_labels = False

    for model_type in MODEL_CHOICES:
        # Check abort flag between models
        if _RESULTS_ABORT["flag"]:
            return JsonResponse({"error": "aborted", "partial": table_rows}, status=499)

        if model_type not in job["results"]:
            continue
        df_err  = pd.read_json(job["results"][model_type]["df_err"], orient="split")
        metrics, hl = compute_metrics_from_error(df_err, DF, cfg)
        has_labels  = hl

        for th_name in TH_NAMES:
            agg     = dict(tp=0, fp=0, tn=0, fn=0, flagged=0, total=0,
                           n_segments=0, n_detected_segs=0)
            per_run = {}
            # Macro-average lists (anomaly runs only) — same as Grid Search
            anom_f1s, anom_precs, anom_recs, anom_accs = [], [], [], []

            for run_id, run_m in metrics.items():
                m = run_m.get(th_name, {})
                per_run[run_id] = m
                agg["flagged"] += m.get("flagged", 0)
                agg["total"]   += 1
                # Anomaly runs only — skip normal (no true anomaly labels)
                if run_id in NORMAL_RUNS:
                    continue
                for k in ["tp", "fp", "tn", "fn"]:
                    agg[k] += m.get(k, 0)
                agg["n_segments"]      += m.get("n_segments", 0)
                agg["n_detected_segs"] += m.get("n_detected_segs", 0)
                # Collect per-run metrics for macro average
                if m.get("f1") is not None:
                    anom_f1s.append(m["f1"])
                    anom_precs.append(m.get("precision", 0))
                    anom_recs.append(m.get("recall", 0))
                    anom_accs.append(m.get("accuracy", 0))

            if has_labels:
                # Macro-average F1/P/R/Acc (mean per-run) — consistent with Grid Search
                mean = lambda lst: round(float(sum(lst)/len(lst)), 4) if lst else 0.0
                agg.update(
                    f1=mean(anom_f1s),
                    precision=mean(anom_precs),
                    recall=mean(anom_recs),
                    accuracy=mean(anom_accs),
                )

            table_rows.append(dict(
                model=model_type, threshold=th_name,
                aggregate=agg, per_run=per_run,
            ))

    return JsonResponse({
        "rows": table_rows, "has_labels": has_labels,
        "runs": ALL_RUNS, "normal_runs": NORMAL_RUNS,
    })


@csrf_exempt
@require_POST
def api_results_abort(request):
    _RESULTS_ABORT["flag"] = True
    return JsonResponse({"aborted": True})
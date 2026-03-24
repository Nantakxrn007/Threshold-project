# dashboard/views/animation.py
"""Animation endpoint — returns pre-computed arrays for JS-driven playback."""
import numpy as np
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ._shared import (
    DF, NORMAL_RUNS, FEATS, LABEL_COLS,
    ThresholdEvaluator, _cfg_from_request,
)
from .single import _get_job, _recompute_err


@require_GET
def api_animation_data(request):
    """
    Returns ALL data for a run as flat arrays so JS can animate frame-by-frame.

    Response shape:
    {
      run_id, n,
      raw:        { conductivity, pH, temperature, voltage },
      error:      [...],
      thresholds: { "P99 Static": [], "Sliding Mu+αStd": [], "Adaptive-z": [], "Entropy-lock": [] },
      labels:     { overall: [], per_sensor: { conductivity, pH, temperature, voltage } },
      has_labels: bool,
      y_range:    { raw_max, err_max }
    }
    """
    job    = _get_job(request)
    run_id = request.GET.get("run", "")
    agg    = request.GET.get("agg", "mean")

    if not job:
        return JsonResponse({"error": "no completed job"}, status=400)
    if not run_id:
        return JsonResponse({"error": "run_id required"}, status=400)

    # ── Raw sensor data ───────────────────────────────────────────────
    df_raw = DF[DF["run_id"] == run_id].reset_index(drop=True)
    if df_raw.empty:
        return JsonResponse({"error": f"run {run_id} not found"}, status=404)

    raw_out = {}
    for feat in FEATS:
        raw_out[feat] = df_raw[feat].fillna(0).tolist() if feat in df_raw.columns else []

    # ── Reconstruction error + thresholds ────────────────────────────
    df_err    = _recompute_err(job, agg)
    dfr       = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
    err_vals  = dfr["overall_error"].fillna(0).tolist()
    n         = len(err_vals)

    # ThresholdEvaluator needs normal baseline
    baseline  = df_err[df_err["run_id"].isin(NORMAL_RUNS)]["overall_error"].dropna().values
    evaluator = ThresholdEvaluator(baseline)
    cfg       = _cfg_from_request(request)
    th1, th2, th3, th4 = evaluator.calculate(dfr["overall_error"], cfg)

    thresholds = {
        "P99 Static":      th1.tolist(),
        "Sliding Mu+αStd": th2.tolist(),
        "Adaptive-z":      th3.tolist(),
        "Entropy-lock":    th4.tolist(),
    }

    # ── Anomaly labels ────────────────────────────────────────────────
    present     = [c for c in LABEL_COLS if c in df_raw.columns]
    has_labels  = bool(present)
    overall_lbl = []
    per_sensor  = {}

    sensor_label_map = {
        "conductivity": "Anomaly C_filled",
        "pH":           "Anomaly P_filled",
        "temperature":  "Anomaly T_filled",
        "voltage":      "Anomaly V_filled",
    }

    if has_labels:
        overall_arr = (df_raw[present].fillna(0).max(axis=1).values > 0).astype(int)
        # Align to error length
        if len(overall_arr) >= n:
            overall_lbl = overall_arr[:n].tolist()
        else:
            overall_lbl = np.concatenate([overall_arr, np.zeros(n - len(overall_arr), int)]).tolist()

        for feat, lbl_col in sensor_label_map.items():
            if lbl_col in df_raw.columns:
                arr = (df_raw[lbl_col].fillna(0).values > 0).astype(int)
                per_sensor[feat] = arr[:n].tolist() if len(arr) >= n else \
                    np.concatenate([arr, np.zeros(n - len(arr), int)]).tolist()
            else:
                per_sensor[feat] = [0] * n

    # ── y-range helpers for JS to set axis domains ────────────────────
    raw_max = max(
        (max(v) if v else 0)
        for v in raw_out.values()
    ) * 1.1 or 1.0
    err_max = max(err_vals) * 1.15 if err_vals else 1.0

    return JsonResponse({
        "run_id":     run_id,
        "n":          n,
        "raw":        raw_out,
        "error":      err_vals,
        "thresholds": thresholds,
        "labels": {
            "overall":    overall_lbl,
            "per_sensor": per_sensor,
        },
        "has_labels": has_labels,
        "y_range": {
            "raw_max": raw_max,
            "err_max": err_max,
        },
    })

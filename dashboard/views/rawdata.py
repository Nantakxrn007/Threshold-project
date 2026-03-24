# dashboard/views/rawdata.py
"""
Raw data API — serve sensor values + anomaly labels per run,
and accept PATCH requests to update labels in memory.
Labels are stored per session in a module-level dict (resets on server restart).
Users can export the patched data as CSV.
"""
import io, json, copy
import numpy as np
import pandas as pd
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from ._shared import DF, ALL_RUNS, FEATS, LABEL_COLS, NORMAL_RUNS, ANOMALY_RUNS

# ── In-memory label store ──────────────────────────────────────────────────────
# { run_id: { label_col: np.ndarray(bool) } }
_LABEL_STORE: dict = {}

# History stack for undo — per run_id
# { run_id: [ snapshot_dict, ... ] }
_LABEL_HISTORY: dict = {}

_MAX_HISTORY = 30


def _get_labels(run_id: str) -> dict:
    """Return current label arrays for a run (initialise from DF if not patched yet)."""
    if run_id not in _LABEL_STORE:
        df_run = DF[DF["run_id"] == run_id].reset_index(drop=True)
        _LABEL_STORE[run_id] = {
            col: (df_run[col].fillna(0).values > 0).astype(int)
            for col in LABEL_COLS
            if col in df_run.columns
        }
    return _LABEL_STORE[run_id]


def _push_history(run_id: str):
    """Save a deep copy of current labels to history."""
    current = _get_labels(run_id)
    snap = {k: v.copy() for k, v in current.items()}
    _LABEL_HISTORY.setdefault(run_id, []).append(snap)
    if len(_LABEL_HISTORY[run_id]) > _MAX_HISTORY:
        _LABEL_HISTORY[run_id].pop(0)


@require_GET
def api_rawdata_run(request):
    """Return sensor values + current labels for a run."""
    run_id = request.GET.get("run", ALL_RUNS[0])
    if run_id not in ALL_RUNS:
        return JsonResponse({"error": "unknown run"}, status=400)

    df_run = DF[DF["run_id"] == run_id].reset_index(drop=True)
    labels = _get_labels(run_id)
    n = len(df_run)

    # Build per-sensor label series
    label_data = {}
    for col in LABEL_COLS:
        if col in labels:
            label_data[col] = labels[col].tolist()
        else:
            label_data[col] = [0] * n

    # Overall anomaly = any sensor flagged
    overall = np.zeros(n, dtype=int)
    for arr in labels.values():
        overall = np.maximum(overall, arr[:n])

    rows = []
    for i in range(n):
        row = {"idx": i}
        for feat in FEATS:
            if feat in df_run.columns:
                v = df_run[feat].iloc[i]
                row[feat] = None if pd.isna(v) else round(float(v), 5)
        for col in LABEL_COLS:
            row[col] = int(label_data[col][i]) if i < len(label_data[col]) else 0
        row["anomaly_any"] = int(overall[i])
        rows.append(row)

    # Segment stats
    def count_segs(arr):
        segs, in_s = 0, False
        for v in arr:
            if v and not in_s: segs += 1; in_s = True
            elif not v: in_s = False
        return segs

    stats = {
        "n_rows": n,
        "n_anomaly_pts": int(overall.sum()),
        "pct_anomaly": round(100 * overall.sum() / max(n, 1), 1),
        "n_segments_overall": count_segs(overall),
        "per_sensor": {
            col: {
                "n_pts": int(sum(label_data[col])),
                "n_segs": count_segs(label_data[col]),
            }
            for col in LABEL_COLS
        },
        "has_history": len(_LABEL_HISTORY.get(run_id, [])),
    }

    return JsonResponse({
        "run_id": run_id,
        "feats": FEATS,
        "label_cols": LABEL_COLS,
        "rows": rows,
        "stats": stats,
        "sensor_data": {
            feat: df_run[feat].where(~df_run[feat].isna(), other=None).tolist()
            for feat in FEATS if feat in df_run.columns
        },
        "label_data": label_data,
        "overall": overall.tolist(),
    })


@csrf_exempt
@require_POST
def api_rawdata_patch(request):
    """
    Patch labels for a run.
    Body: { run_id, label_col, start_idx, end_idx, value (0|1) }
    Or:   { run_id, label_col, indices: [...], value }
    """
    p = json.loads(request.body)
    run_id    = p.get("run_id")
    label_col = p.get("label_col")
    value     = int(p.get("value", 1))

    if run_id not in ALL_RUNS:
        return JsonResponse({"error": "unknown run"}, status=400)
    if label_col not in LABEL_COLS:
        return JsonResponse({"error": "unknown label column"}, status=400)

    _push_history(run_id)
    labels = _get_labels(run_id)

    if label_col not in labels:
        n = len(DF[DF["run_id"] == run_id])
        labels[label_col] = np.zeros(n, dtype=int)

    arr = labels[label_col]

    # Range or explicit indices
    if "indices" in p:
        for idx in p["indices"]:
            if 0 <= idx < len(arr):
                arr[idx] = value
    else:
        s = max(0, int(p.get("start_idx", 0)))
        e = min(len(arr) - 1, int(p.get("end_idx", s)))
        arr[s:e + 1] = value

    # Recount stats
    overall = np.zeros(len(arr), dtype=int)
    for a in labels.values():
        overall = np.maximum(overall, a)

    return JsonResponse({
        "ok": True,
        "n_anomaly_pts": int(overall.sum()),
        "label_data": {col: labels[col].tolist() for col in labels},
        "overall": overall.tolist(),
    })


@csrf_exempt
@require_POST
def api_rawdata_undo(request):
    """Undo last label patch for a run."""
    p = json.loads(request.body)
    run_id = p.get("run_id")
    hist = _LABEL_HISTORY.get(run_id, [])
    if not hist:
        return JsonResponse({"error": "nothing to undo"}, status=400)

    snap = hist.pop()
    _LABEL_STORE[run_id] = {k: v.copy() for k, v in snap.items()}
    labels = _LABEL_STORE[run_id]
    overall = np.zeros(max(len(v) for v in labels.values()), dtype=int)
    for a in labels.values():
        overall = np.maximum(overall, a)

    return JsonResponse({
        "ok": True,
        "label_data": {col: labels[col].tolist() for col in labels},
        "overall": overall.tolist(),
        "history_remaining": len(hist),
    })


@csrf_exempt
@require_POST
def api_rawdata_reset(request):
    """Reset labels for a run back to original DF values."""
    p = json.loads(request.body)
    run_id = p.get("run_id")
    if run_id in _LABEL_STORE:
        del _LABEL_STORE[run_id]
    if run_id in _LABEL_HISTORY:
        del _LABEL_HISTORY[run_id]
    return JsonResponse({"ok": True})


@require_GET
def api_rawdata_export(request):
    """Export patched labels — ?run=X for one run, ?run=all for all runs."""
    run_param = request.GET.get("run", "all")
    runs = ALL_RUNS if run_param == "all" else [r for r in [run_param] if r in ALL_RUNS]

    frames = []
    for run_id in runs:
        df_run = DF[DF["run_id"] == run_id].reset_index(drop=True).copy()
        labels = _get_labels(run_id)
        for col, arr in labels.items():
            n = len(df_run)
            padded = np.zeros(n, dtype=int)
            padded[:min(n, len(arr))] = arr[:min(n, len(arr))]
            df_run[col] = padded
        frames.append(df_run)

    df_out = pd.concat(frames, ignore_index=True)

    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)

    fname = "all_runs_labels.csv" if run_param == "all" else f"{runs[0]}_labels.csv"
    resp = HttpResponse(buf.read(), content_type="text/csv")
    resp["Content-Disposition"] = f'attachment; filename="{fname}"'
    return resp
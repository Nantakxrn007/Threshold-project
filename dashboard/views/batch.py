# dashboard/views/batch.py
"""Train-All endpoints: training, status, PCA grid, error inspector."""
import json, threading, uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

# Stop flags per job — set via api_stop_batch
STOP_FLAGS: dict = {}

from ._shared import (
    DF, BATCH_JOBS, NORMAL_RUNS, ANOMALY_RUNS, ALL_RUNS,
    MODEL_CHOICES, FEATS, TH_COLORS, _LAYOUT, _to_json,
    Pipeline, ThresholdEvaluator, _add_anomaly_marks,
)


def _recompute_err_batch(res: dict, agg: str) -> "pd.DataFrame":
    """Recompute overall_error with given agg from per-sensor columns in batch result."""
    cache_key = f"df_err_{agg}"
    if res.get(cache_key):
        return pd.read_json(res[cache_key], orient="split")
    df = pd.read_json(res["df_err"], orient="split")
    if agg == "max":
        df["overall_error"] = df[FEATS].max(axis=1)
    elif agg == "l2":
        df["overall_error"] = np.sqrt((df[FEATS] ** 2).sum(axis=1))
    else:
        df["overall_error"] = df[FEATS].mean(axis=1)
    res[cache_key] = df.to_json(orient="split")
    return df


@csrf_exempt
@require_POST
def api_train_all(request):
    params = json.loads(request.body)
    job_id = uuid.uuid4().hex[:8]
    progress = {m: {"status": "pending", "epoch": 0, "loss": None} for m in MODEL_CHOICES}
    BATCH_JOBS[job_id] = {
        "status": "training",
        "epochs": int(params.get("epochs", 30)),
        "progress": progress,
        "results": {},
        "error": None,
    }

    def _worker():
        try:
            df_normal = DF[DF["run_id"].isin(NORMAL_RUNS)]
            for model_type in MODEL_CHOICES:
                # Check stop flag
                if STOP_FLAGS.get(job_id):
                    BATCH_JOBS[job_id]["status"] = "stopped"
                    STOP_FLAGS.pop(job_id, None)
                    return
                BATCH_JOBS[job_id]["progress"][model_type]["status"] = "training"
                pipe = Pipeline(
                    model_type=model_type,
                    seq_len=int(params.get("seq_len", 10)),
                    ewma=float(params.get("ewma", 0.3)),
                    epochs=int(params.get("epochs", 30)),
                    batch_size=int(params.get("batch_size", 64)),
                    hidden=int(params.get("hidden", 16)),
                    layers=int(params.get("layers", 1)),
                    lr=float(params.get("lr", 0.005)),
                )

                def _cb(ep, loss, mt=model_type):
                    BATCH_JOBS[job_id]["progress"][mt]["epoch"] = ep
                    BATCH_JOBS[job_id]["progress"][mt]["loss"]  = round(loss, 6)

                pipe.train(df_normal, on_epoch=_cb)
                df_err, z_vals, z_labs = pipe.evaluate(DF)
                emb = PCA(n_components=2, random_state=42).fit_transform(z_vals)
                BATCH_JOBS[job_id]["results"][model_type] = {
                    "df_err":    df_err.to_json(orient="split"),
                    "z_vals":    z_vals.tolist(),
                    "z_labs":    z_labs,
                    "pca_cache": emb.tolist(),
                }
                BATCH_JOBS[job_id]["progress"][model_type]["status"] = "done"
            BATCH_JOBS[job_id]["status"] = "done"
        except Exception as exc:
            BATCH_JOBS[job_id].update({"status": "error", "error": str(exc)})

    threading.Thread(target=_worker, daemon=True).start()
    return JsonResponse({"job_id": job_id})


@require_GET
def api_status_all(request, job_id):
    job = BATCH_JOBS.get(job_id)
    if not job:
        return JsonResponse({"error": "job not found"}, status=404)
    return JsonResponse({
        "status": job["status"], "epochs": job["epochs"],
        "progress": job["progress"], "error": job.get("error"),
    })


def _get_batch_job(request):
    job_id = request.GET.get("job_id")
    if job_id and job_id in BATCH_JOBS and BATCH_JOBS[job_id]["status"] == "done":
        return BATCH_JOBS[job_id]
    for jid in reversed(list(BATCH_JOBS)):
        if BATCH_JOBS[jid]["status"] == "done":
            return BATCH_JOBS[jid]
    return None


@require_GET
def api_chart_model_umap(request):
    job        = _get_batch_job(request)
    model_type = request.GET.get("model", MODEL_CHOICES[0])
    color_by   = request.GET.get("color_by", "run")
    if not job or model_type not in job["results"]:
        return JsonResponse({"error": "not ready"}, status=400)

    res    = job["results"][model_type]
    emb    = np.array(res["pca_cache"])
    z_labs = res["z_labs"]
    fig    = go.Figure()

    if color_by == "run":
        for run in ALL_RUNS:
            idx = np.array([i for i, l in enumerate(z_labs) if l == run])
            if not len(idx): continue
            clr = "#16a34a" if run in NORMAL_RUNS else "#dc2626"
            sym = "circle"  if run in NORMAL_RUNS else "diamond"
            fig.add_trace(go.Scatter(
                x=emb[idx, 0].tolist(), y=emb[idx, 1].tolist(),
                mode="markers", name=run,
                marker=dict(size=5, color=clr, opacity=0.75, symbol=sym),
            ))
    else:
        for grp, clr, sym in [("Normal", "#16a34a", "circle"), ("Anomaly", "#dc2626", "diamond")]:
            mask = [(l in NORMAL_RUNS) == (grp == "Normal") for l in z_labs]
            idx  = np.where(mask)[0]
            if not len(idx): continue
            fig.add_trace(go.Scatter(
                x=emb[idx, 0].tolist(), y=emb[idx, 1].tolist(),
                mode="markers", name=grp,
                marker=dict(size=5, color=clr, opacity=0.75, symbol=sym),
            ))

    fig.update_layout(**_LAYOUT, height=300,
        title=dict(text=f"PCA — {model_type}", font=dict(size=12, color="#0f172a")))
    return JsonResponse(_to_json(fig))


@require_GET
def api_chart_model_error(request):
    job         = _get_batch_job(request)
    model_type  = request.GET.get("model", MODEL_CHOICES[0])
    run_id      = request.GET.get("run", ANOMALY_RUNS[0])
    agg         = request.GET.get("agg", "mean")
    show_labels = request.GET.get("show_labels", "1") == "1"
    if not job or model_type not in job["results"]:
        return JsonResponse({"error": "not ready"}, status=400)

    cfg = {
        "th1_pct":    float(request.GET.get("th1_pct",    99.0)),
        "th1_mode":         request.GET.get("th1_mode",  "sliding"),
        "th1_win":    float(request.GET.get("th1_win",   100)),
        "th1_recalc": float(request.GET.get("th1_recalc", 10)),
        "th2_alpha":  float(request.GET.get("th2_alpha",   3.5)),
        "th2_win":    float(request.GET.get("th2_win",    80)),
        "th2_recalc": float(request.GET.get("th2_recalc", 50)),
        "th3_zmin":   float(request.GET.get("th3_zmin",   2.0)),
        "th3_zmax":   float(request.GET.get("th3_zmax",  10.0)),
        "th3_win":    float(request.GET.get("th3_win",    80)),
        "th3_recalc": float(request.GET.get("th3_recalc",  1)),
        "th4_alpha":  float(request.GET.get("th4_alpha",   3.5)),
        "th4_win":    float(request.GET.get("th4_win",   150)),
        "th4_cons":   float(request.GET.get("th4_cons",    5)),
        "th4_eth":    float(request.GET.get("th4_eth",   0.95)),
        "th4_recalc": float(request.GET.get("th4_recalc",  1)),
    }
    res       = job["results"][model_type]
    df_err    = _recompute_err_batch(res, agg)
    evaluator = ThresholdEvaluator(df_err[df_err["run_id"].isin(NORMAL_RUNS)]["overall_error"].dropna().values)
    dfr       = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
    df_raw    = DF[DF["run_id"] == run_id].reset_index(drop=True)
    th1, th2, th3, th4 = evaluator.calculate(dfr["overall_error"], cfg)
    err_vals  = dfr["overall_error"].fillna(0).values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfr.index.tolist(), y=dfr["overall_error"].tolist(),
        mode="lines", name="Error", line=dict(color="#0f172a", width=2),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.04)",
    ))
    for name, vals, dash in [
        ("P99 Static", th1, "dot"), ("Sliding Mu+αStd", th2, "dash"),
        ("Adaptive-z", th3, "longdash"), ("Entropy-lock", th4, "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=dfr.index.tolist(), y=vals.tolist(),
            mode="lines", name=name,
            line=dict(color=TH_COLORS[name], dash=dash, width=1.6),
        ))
    _TH_BAND = {
        "P99 Static":      ("rgba(234,179,8,0.18)",    "rgba(234,179,8,0.65)",   "#ca8a04"),
        "Sliding Mu+αStd": ("rgba(59,130,246,0.13)",  "rgba(59,130,246,0.55)",  "#3b82f6"),
        "Adaptive-z":      ("rgba(16,185,129,0.13)",  "rgba(16,185,129,0.55)",  "#10b981"),
        "Entropy-lock":    ("rgba(124,58,237,0.13)",  "rgba(124,58,237,0.55)",  "#7c3aed"),
    }
    y_max = float(dfr["overall_error"].max()) * 1.15 or 1.0
    for _thn, _thv in [("P99 Static",th1),("Sliding Mu+αStd",th2),
                        ("Adaptive-z",th3),("Entropy-lock",th4)]:
        fill_rgba, marker_rgba, line_hex = _TH_BAND[_thn]
        _anom = np.where(err_vals > _thv)[0]
        band_group = f"band_{_thn}"
        if not len(_anom):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                name=f"{_thn} band",
                legendgroup=band_group,
                showlegend=True,
                marker=dict(size=10, color=marker_rgba, symbol="square",
                            line=dict(width=1, color=line_hex)),
                hoverinfo="skip",
            ))
            continue
        segs = np.split(_anom, np.where(np.diff(_anom)!=1)[0]+1)
        for i, _g in enumerate(segs):
            if len(_g):
                x0, x1 = int(_g[0])-0.5, int(_g[-1])+0.5
                fig.add_trace(go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[0, 0, y_max, y_max, 0],
                    mode="lines", fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(width=0),
                    name=f"{_thn} band",
                    legendgroup=band_group,
                    showlegend=(i == 0),
                    legendrank=1000,
                    marker=dict(color=marker_rgba, symbol="square",
                                line=dict(width=1, color=line_hex)),
                    hoverinfo="skip",
                ))

    if show_labels:
        fig = _add_anomaly_marks(fig, df_raw, show_raw=False)

    fig.update_layout(**_LAYOUT, height=260,
        title=dict(text=f"{model_type} — {run_id} ({agg.upper()})", font=dict(size=12, color="#0f172a")))
    return JsonResponse(_to_json(fig))


@csrf_exempt
@require_POST
def api_clear_batch_cache(request):
    """ลบ batch jobs ทั้งหมดออกจาก memory — คืน RAM และทำให้ตอบสนองเร็วขึ้น"""
    count = len(BATCH_JOBS)
    BATCH_JOBS.clear()
    return JsonResponse({"cleared": count, "message": f"Cleared {count} batch job(s) from memory"})


@require_GET
def api_device_info(request):
    """บอก device ที่ใช้ train — CUDA / MPS / CPU"""
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        name   = torch.cuda.get_device_name(0)
        mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        detail = f"{name} ({mem_gb} GB)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        detail = "Apple Silicon (MPS)"
    else:
        device = "cpu"
        import os
        detail = f"{os.cpu_count()} cores"
    return JsonResponse({"device": device, "detail": detail})


@csrf_exempt
@require_POST
def api_stop_batch(request):
    """Stop a running batch job and clear its results from memory."""
    p      = json.loads(request.body) if request.body else {}
    job_id = p.get("job_id") or p.get("job") or ""
    # Set stop flag — worker checks between models
    if job_id and job_id in BATCH_JOBS:
        STOP_FLAGS[job_id] = True
        BATCH_JOBS[job_id]["status"] = "stopping"
    # Also clear all batch jobs to free RAM
    cleared = len(BATCH_JOBS)
    BATCH_JOBS.clear()
    STOP_FLAGS.clear()
    return JsonResponse({"ok": True, "cleared": cleared})
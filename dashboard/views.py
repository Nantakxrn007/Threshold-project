# dashboard/views.py
import json
import threading
import uuid

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import umap.umap_ as umap_lib
from sklearn.decomposition import PCA  # <-- เพิ่ม PCA ตรงนี้
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .ml import (
    ALL_RUNS, ANOMALY_RUNS, FEATS, NORMAL_RUNS, MODEL_MAP, LABEL_COLS,
    Pipeline, ThresholdEvaluator, get_training_data, compute_metrics_from_error,
)

# ── Globals ──────────────────────────────────────────────────────────────────

DF = get_training_data()   # โหลดครั้งเดียวตอน import

JOBS: dict = {}
BATCH_JOBS: dict = {}   # Train-All jobs

MODEL_CHOICES = [
    "LSTM-AE", "GRU-AE", "CNN-LSTM-AE",
    "Plain-AE", "Plain-LSTM", "LSTM-Attention-AE",
]

# ── Plotly white theme ────────────────────────────────────────────────────────

_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, system-ui, sans-serif", color="#64748b", size=11),
    xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", color="#94a3b8", zeroline=False),
    yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", color="#94a3b8", zeroline=False),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#e2e8f0",
        font=dict(color="#0f172a", family="Inter, system-ui, sans-serif"),
    ),
    margin=dict(l=50, r=16, t=44, b=36),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1,
        font=dict(size=10),
    ),
)

FEAT_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899"]

TH_COLORS = {
    "P99 Static":      "#ef4444",
    "Sliding Mu+αStd": "#3b82f6",
    "Adaptive-z":      "#10b981",
    "Entropy-lock":    "#7c3aed",
}

def _to_json(fig: go.Figure) -> dict:
    """Serialize Plotly figure → JSON-safe dict."""
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

# ── Pages ─────────────────────────────────────────────────────────────────────

def index(request):
    # แก้บัค json.dumps ตัวอักษรแตก ด้วยการส่งเป็น List ปกติ
    return render(request, "dashboard/index.html", {
        "all_runs":      ALL_RUNS,
        "normal_runs":   NORMAL_RUNS,
        "anomaly_runs":  ANOMALY_RUNS,
        "model_choices": MODEL_CHOICES,
    })

# ── Training API ──────────────────────────────────────────────────────────────

@csrf_exempt
@require_POST
def api_train(request):
    params = json.loads(request.body)
    job_id = uuid.uuid4().hex[:8]

    JOBS[job_id] = {
        "status":     "training",
        "loss":       [],
        "epochs":     int(params.get("epochs", 30)),
        "error":      None,
        "df_err":     None,
        "z_vals":     None,
        "z_labs":     None,
        "umap_cache": None,
        "umap_params": None,
    }

    def _worker():
        try:
            pipe = Pipeline(
                model_type = params["model_type"],
                seq_len    = int(params["seq_len"]),
                ewma       = float(params["ewma"]),
                epochs     = int(params["epochs"]),
                batch_size = int(params["batch_size"]),
                hidden     = int(params["hidden"]),
                layers     = int(params["layers"]),
                lr         = float(params["lr"]),
            )
            df_normal = DF[DF["run_id"].isin(NORMAL_RUNS)]
            pipe.train(df_normal)
            JOBS[job_id]["loss"] = pipe.loss_history

            df_err, z_vals, z_labs = pipe.evaluate(DF)

            # Pre-compute ด้วย PCA (เพราะเร็วกว่า UMAP มาก) เป็นค่า Default แรกสุด
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(z_vals)

            JOBS[job_id].update({
                "status":      "done",
                "df_err":      df_err.to_json(orient="split"),
                "z_vals":      z_vals.tolist(),
                "z_labs":      z_labs,
                "umap_cache":  emb.tolist(),
                "umap_params": "pca",  # เก็บ key ว่าตอนนี้ใช้ PCA อยู่
            })
        except Exception as exc:
            JOBS[job_id].update({"status": "error", "error": str(exc)})

    threading.Thread(target=_worker, daemon=True).start()
    return JsonResponse({"job_id": job_id})

@require_GET
def api_status(request, job_id):
    job = JOBS.get(job_id)
    if not job:
        return JsonResponse({"error": "job not found"}, status=404)
    return JsonResponse({
        "status": job["status"],
        "loss":   job["loss"],
        "epochs": job["epochs"],
        "error":  job.get("error"),
    })

# ── Chart helpers ─────────────────────────────────────────────────────────────

def _get_job(request) -> dict | None:
    job_id = request.GET.get("job_id")
    if job_id and job_id in JOBS and JOBS[job_id]["status"] == "done":
        return JOBS[job_id]
    for jid in reversed(list(JOBS)):
        if JOBS[jid]["status"] == "done":
            return JOBS[jid]
    return None

# ── Chart APIs ────────────────────────────────────────────────────────────────

@require_GET
def api_chart_umap(request):
    job = _get_job(request)
    if not job:
        return JsonResponse({"error": "no completed job"}, status=400)

    method      = request.GET.get("method", "pca")  # รับค่า method ว่าจะเอา pca หรือ umap
    n_neighbors = int(request.GET.get("nn", 15))
    min_dist    = float(request.GET.get("md", 0.1))
    color_by    = request.GET.get("color_by", "run")
    
    # สร้าง Cache Key
    cache_key   = "pca" if method == "pca" else f"umap_{n_neighbors}_{min_dist}"

    z_vals = np.array(job["z_vals"])
    z_labs = job["z_labs"]

    # Recompute ถ้าระบุพารามิเตอร์ใหม่ หรือเปลี่ยนจาก PCA เป็น UMAP
    if job.get("umap_params") != cache_key:
        if method == "pca":
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(z_vals)
        else:
            reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            emb     = reducer.fit_transform(z_vals)
            
        job["umap_cache"]  = emb.tolist()
        job["umap_params"] = cache_key
    else:
        emb = np.array(job["umap_cache"])

    emb = np.array(emb)
    fig = go.Figure()

    if color_by == "run":
        for run in ALL_RUNS:
            idx  = np.array([i for i, l in enumerate(z_labs) if l == run])
            clr  = "#16a34a" if run in NORMAL_RUNS else "#dc2626"
            sym  = "circle" if run in NORMAL_RUNS else "diamond"
            if not len(idx):
                continue
            fig.add_trace(go.Scatter(
                x=emb[idx, 0].tolist(), y=emb[idx, 1].tolist(),
                mode="markers", name=run,
                marker=dict(size=5, color=clr, opacity=0.75, symbol=sym),
                hovertemplate=f"{run}<extra></extra>",
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
                hovertemplate=f"{grp}<extra></extra>",
            ))

    title_text = "PCA Projection (Fast)" if method == "pca" else "UMAP Projection (Detailed)"
    fig.update_layout(
        **_LAYOUT, height=460,
        title=dict(text=title_text, font=dict(size=13, color="#0f172a")),
    )
    return JsonResponse(_to_json(fig))

@require_GET
def api_chart_raw(request):
    run_id = request.GET.get("run", ALL_RUNS[0])
    dfr    = DF[DF["run_id"] == run_id].reset_index(drop=True)

    fig = go.Figure()
    for feat, clr in zip(FEATS, FEAT_COLORS):
        fig.add_trace(go.Scatter(
            x=dfr.index.tolist(), y=dfr[feat].tolist(),
            mode="lines", name=feat,
            line=dict(color=clr, width=1.5),
            hovertemplate=f"{feat}=%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        **_LAYOUT, height=230,
        title=dict(text=f"Raw Sensor — {run_id}", font=dict(size=12, color="#0f172a")),
    )
    return JsonResponse(_to_json(fig))

@require_GET
def api_chart_error(request):
    job    = _get_job(request)
    run_id = request.GET.get("run", ANOMALY_RUNS[0])
    if not job:
        return JsonResponse({"error": "no completed job"}, status=400)

    df_err = pd.read_json(job["df_err"], orient="split")
    dfr    = df_err[df_err["run_id"] == run_id].reset_index(drop=True)

    fig = go.Figure()
    for feat, clr in zip(FEATS, FEAT_COLORS):
        if feat in dfr.columns:
            fig.add_trace(go.Scatter(
                x=dfr.index.tolist(), y=dfr[feat].tolist(),
                mode="lines", name=feat,
                line=dict(color=clr, width=1, dash="dot"), opacity=0.6,
                hovertemplate=f"{feat}=%{{y:.4f}}<extra></extra>",
            ))
    fig.add_trace(go.Scatter(
        x=dfr.index.tolist(), y=dfr["overall_error"].tolist(),
        mode="lines", name="Overall",
        line=dict(color="#0f172a", width=2.5),
        hovertemplate="Overall=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT, height=230,
        title=dict(text="Reconstruction Error", font=dict(size=12, color="#0f172a")),
    )
    return JsonResponse(_to_json(fig))

@require_GET
def api_chart_inspector_umap(request):
    job    = _get_job(request)
    run_id = request.GET.get("run", ANOMALY_RUNS[0])
    if not job or not job.get("umap_cache"):
        return JsonResponse({"error": "no completed job"}, status=400)

    z_labs = job["z_labs"]
    emb    = np.array(job["umap_cache"])

    fig = go.Figure()
    for grp, clr, sz, op in [
        ("other_normal",  "#bbf7d0", 5, 0.5),
        ("other_anomaly", "#fecaca", 5, 0.5),
        ("selected",      "#2563eb", 7, 0.95),
    ]:
        if grp == "selected":
            mask = np.array([l == run_id for l in z_labs])
        elif grp == "other_normal":
            mask = np.array([(l != run_id) and (l in NORMAL_RUNS) for l in z_labs])
        else:
            mask = np.array([(l != run_id) and (l in ANOMALY_RUNS) for l in z_labs])
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        fig.add_trace(go.Scatter(
            x=emb[idx, 0].tolist(), y=emb[idx, 1].tolist(),
            mode="markers", name=grp.replace("_", " "),
            marker=dict(size=sz, color=clr, opacity=op),
            customdata=[z_labs[i] for i in idx],
            hovertemplate="%{customdata}<extra></extra>",
        ))

    method_name = "PCA" if job.get("umap_params") == "pca" else "UMAP"
    fig.update_layout(
        **_LAYOUT, height=300,
        title=dict(text=f"{method_name} — {run_id} highlighted", font=dict(size=12, color="#0f172a")),
    )
    return JsonResponse(_to_json(fig))

@require_GET
def api_chart_threshold(request):
    job    = _get_job(request)
    run_id = request.GET.get("run", ANOMALY_RUNS[0])
    if not job:
        return JsonResponse({"error": "no completed job"}, status=400)

    cfg = {
        "th1_pct":    float(request.GET.get("th1_pct",    99.0)),
        "th2_alpha":  float(request.GET.get("th2_alpha",   3.5)),
        "th2_win":    float(request.GET.get("th2_win",    80)),
        "th2_recalc": float(request.GET.get("th2_recalc", 50)),
        "th3_zmin":   float(request.GET.get("th3_zmin",   2.0)),
        "th3_zmax":   float(request.GET.get("th3_zmax",  10.0)),
        "th3_win":    float(request.GET.get("th3_win",    80)),
        "th4_alpha":  float(request.GET.get("th4_alpha",   3.5)),
        "th4_win":    float(request.GET.get("th4_win",   150)),
        "th4_cons":   float(request.GET.get("th4_cons",    5)),
        "th4_eth":    float(request.GET.get("th4_eth",    0.95)),
    }

    df_err    = pd.read_json(job["df_err"], orient="split")
    df_norm   = df_err[df_err["run_id"].isin(NORMAL_RUNS)]
    evaluator = ThresholdEvaluator(df_norm["overall_error"].dropna().values)

    dfr       = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
    th1, th2, th3, th4 = evaluator.calculate(dfr["overall_error"], cfg)
    err_vals  = dfr["overall_error"].fillna(0).values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfr.index.tolist(), y=dfr["overall_error"].tolist(),
        mode="lines", name="Overall Error",
        line=dict(color="#0f172a", width=2.5),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.03)",
        hovertemplate="Error=%{y:.4f}<extra></extra>",
    ))
    for name, vals, dash in [
        ("P99 Static",      th1, "dot"),
        ("Sliding Mu+αStd", th2, "dash"),
        ("Adaptive-z",      th3, "longdash"),
        ("Entropy-lock",    th4, "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=dfr.index.tolist(), y=vals.tolist(),
            mode="lines", name=name,
            line=dict(color=TH_COLORS[name], dash=dash, width=1.8),
            hovertemplate=f"{name}=%{{y:.4f}}<extra></extra>",
        ))

    anom = np.where(err_vals > th1)[0]
    if len(anom):
        groups = np.split(anom, np.where(np.diff(anom) != 1)[0] + 1)
        for g in groups:
            if len(g):
                fig.add_vrect(
                    x0=int(g[0]), x1=int(g[-1]),
                    fillcolor="rgba(239,68,68,0.07)",
                    layer="below", line_width=0,
                )

    height = int(request.GET.get("height", 400))
    fig.update_layout(
        **_LAYOUT, height=height,
        title=dict(text=f"Thresholds — {run_id}", font=dict(size=12, color="#0f172a")),
    )

    stats = {}
    for name, vals in [
        ("P99 Static", th1), ("Sliding Mu+αStd", th2),
        ("Adaptive-z", th3), ("Entropy-lock", th4),
    ]:
        n_flag    = int((err_vals > vals).sum())
        stats[name] = {
            "flagged": n_flag,
            "pct":     round(100 * n_flag / max(len(err_vals), 1), 1),
        }

    result       = _to_json(fig)
    result["stats"] = stats
    return JsonResponse(result)

# ═══════════════════════════════════════════════════════════════════════════════
# ── Train-All API ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def api_train_all(request):
    params = json.loads(request.body)
    job_id = uuid.uuid4().hex[:8]

    progress = {m: {"status": "pending", "epoch": 0, "loss": None} for m in MODEL_CHOICES}
    BATCH_JOBS[job_id] = {
        "status":   "training",
        "epochs":   int(params.get("epochs", 30)),
        "progress": progress,
        "results":  {},
        "error":    None,
    }

    def _worker():
        try:
            df_normal = DF[DF["run_id"].isin(NORMAL_RUNS)]
            for model_type in MODEL_CHOICES:
                BATCH_JOBS[job_id]["progress"][model_type]["status"] = "training"

                pipe = Pipeline(
                    model_type = model_type,
                    seq_len    = int(params.get("seq_len", 10)),
                    ewma       = float(params.get("ewma", 0.3)),
                    epochs     = int(params.get("epochs", 30)),
                    batch_size = int(params.get("batch_size", 64)),
                    hidden     = int(params.get("hidden", 16)),
                    layers     = int(params.get("layers", 1)),
                    lr         = float(params.get("lr", 0.005)),
                )

                def _cb(ep, loss, mt=model_type):
                    BATCH_JOBS[job_id]["progress"][mt]["epoch"] = ep
                    BATCH_JOBS[job_id]["progress"][mt]["loss"]  = round(loss, 6)

                pipe.train(df_normal, on_epoch=_cb)
                df_err, z_vals, z_labs = pipe.evaluate(DF)

                # PCA (fast default)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                emb = pca.fit_transform(z_vals)

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
        "status":   job["status"],
        "epochs":   job["epochs"],
        "progress": job["progress"],
        "error":    job.get("error"),
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
    """PCA scatter for one model in the Train-All job."""
    job        = _get_batch_job(request)
    model_type = request.GET.get("model", MODEL_CHOICES[0])
    color_by   = request.GET.get("color_by", "run")
    if not job or model_type not in job["results"]:
        return JsonResponse({"error": "not ready"}, status=400)

    res    = job["results"][model_type]
    emb    = np.array(res["pca_cache"])
    z_labs = res["z_labs"]

    fig = go.Figure()
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
                hovertemplate=f"{run}<extra></extra>",
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
                hovertemplate=f"{grp}<extra></extra>",
            ))

    fig.update_layout(**_LAYOUT, height=300,
        title=dict(text=f"PCA — {model_type}", font=dict(size=12, color="#0f172a")))
    return JsonResponse(_to_json(fig))


@require_GET
def api_chart_model_error(request):
    """Error + threshold chart for one model + run in Train-All."""
    job        = _get_batch_job(request)
    model_type = request.GET.get("model", MODEL_CHOICES[0])
    run_id     = request.GET.get("run", ANOMALY_RUNS[0])
    if not job or model_type not in job["results"]:
        return JsonResponse({"error": "not ready"}, status=400)

    cfg = {
        "th1_pct":   float(request.GET.get("th1_pct",   99.0)),
        "th2_alpha": float(request.GET.get("th2_alpha",  3.5)),
        "th2_win":   float(request.GET.get("th2_win",   80)),
        "th2_recalc":float(request.GET.get("th2_recalc",50)),
        "th3_zmin":  float(request.GET.get("th3_zmin",  2.0)),
        "th3_zmax":  float(request.GET.get("th3_zmax", 10.0)),
        "th3_win":   float(request.GET.get("th3_win",  80)),
        "th4_alpha": float(request.GET.get("th4_alpha", 3.5)),
        "th4_win":   float(request.GET.get("th4_win",  150)),
        "th4_cons":  float(request.GET.get("th4_cons",   5)),
        "th4_eth":   float(request.GET.get("th4_eth",  0.95)),
    }

    df_err   = pd.read_json(job["results"][model_type]["df_err"], orient="split")
    df_norm  = df_err[df_err["run_id"].isin(NORMAL_RUNS)]
    evaluator = ThresholdEvaluator(df_norm["overall_error"].dropna().values)

    dfr      = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
    th1, th2, th3, th4 = evaluator.calculate(dfr["overall_error"], cfg)
    err_vals = dfr["overall_error"].fillna(0).values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfr.index.tolist(), y=dfr["overall_error"].tolist(),
        mode="lines", name="Error",
        line=dict(color="#0f172a", width=2),
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

    # shade anomaly regions
    anom = np.where(err_vals > th1)[0]
    if len(anom):
        for g in np.split(anom, np.where(np.diff(anom) != 1)[0] + 1):
            if len(g):
                fig.add_vrect(x0=int(g[0]), x1=int(g[-1]),
                    fillcolor="rgba(239,68,68,0.07)", layer="below", line_width=0)

    fig.update_layout(**_LAYOUT, height=260,
        title=dict(text=f"{model_type} — {run_id}", font=dict(size=12, color="#0f172a")))
    return JsonResponse(_to_json(fig))


@require_GET
def api_results_table(request):
    """Full confusion matrix table: all models × all thresholds × all runs."""
    job = _get_batch_job(request)
    if not job:
        return JsonResponse({"error": "no completed batch job"}, status=400)

    cfg = {
        "th1_pct":   float(request.GET.get("th1_pct",   99.0)),
        "th2_alpha": float(request.GET.get("th2_alpha",  3.5)),
        "th2_win":   float(request.GET.get("th2_win",   80)),
        "th2_recalc":float(request.GET.get("th2_recalc",50)),
        "th3_zmin":  float(request.GET.get("th3_zmin",  2.0)),
        "th3_zmax":  float(request.GET.get("th3_zmax", 10.0)),
        "th3_win":   float(request.GET.get("th3_win",  80)),
        "th4_alpha": float(request.GET.get("th4_alpha", 3.5)),
        "th4_win":   float(request.GET.get("th4_win",  150)),
        "th4_cons":  float(request.GET.get("th4_cons",   5)),
        "th4_eth":   float(request.GET.get("th4_eth",  0.95)),
    }

    table_rows = []
    has_labels = False

    for model_type in MODEL_CHOICES:
        if model_type not in job["results"]:
            continue
        df_err = pd.read_json(job["results"][model_type]["df_err"], orient="split")
        metrics, hl = compute_metrics_from_error(df_err, DF, cfg)
        has_labels = hl

        for th_name in ["P99 Static", "Sliding Mu+αStd", "Adaptive-z", "Entropy-lock"]:
            # aggregate across all runs
            agg = dict(tp=0, fp=0, tn=0, fn=0, flagged=0, total=0, n_segments=0, n_detected_segs=0)
            per_run = {}
            for run_id, run_m in metrics.items():
                m = run_m.get(th_name, {})
                per_run[run_id] = m
                for k in ["tp","fp","tn","fn","flagged","n_segments","n_detected_segs"]:
                    agg[k] += m.get(k, 0)
                agg["total"] += 1

            # compute aggregate metrics
            if has_labels:
                prec = agg["tp"] / max(agg["tp"] + agg["fp"], 1)
                rec  = agg["tp"] / max(agg["tp"] + agg["fn"], 1)
                f1   = 2*prec*rec / max(prec+rec, 1e-9)
                acc  = (agg["tp"]+agg["tn"]) / max(sum(agg[k] for k in ["tp","fp","tn","fn"]), 1)
                agg.update(precision=round(prec,4), recall=round(rec,4),
                            f1=round(f1,4), accuracy=round(acc,4))

            table_rows.append(dict(
                model=model_type, threshold=th_name,
                aggregate=agg, per_run=per_run,
            ))

    return JsonResponse({"rows": table_rows, "has_labels": has_labels,
                         "runs": ALL_RUNS, "normal_runs": NORMAL_RUNS})
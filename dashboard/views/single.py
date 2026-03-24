# dashboard/views/single.py
"""Single-model train / status / chart endpoints."""
import threading, uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import umap.umap_ as umap_lib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from ._shared import (
    DF, JOBS, NORMAL_RUNS, ANOMALY_RUNS, ALL_RUNS, FEATS,
    FEAT_COLORS, TH_COLORS, _LAYOUT, _to_json,
    Pipeline, ThresholdEvaluator, _add_anomaly_marks,
)


def _recompute_err(job: dict, agg: str) -> "pd.DataFrame":
    """Return df_err recomputed with the requested agg method.
    Uses a small per-job cache keyed by agg to avoid re-running evaluate."""
    import pandas as pd
    cache_key = f"df_err_{agg}"
    if job.get(cache_key):
        return pd.read_json(job[cache_key], orient="split")

    # Rebuild from stored z_vals / predictions is not possible without model —
    # instead we recalculate overall_error from the per-sensor columns already in df_err.
    df = pd.read_json(job["df_err"], orient="split")
    if agg == "max":
        df["overall_error"] = df[FEATS].max(axis=1)
    elif agg == "l2":
        import numpy as np
        df["overall_error"] = np.sqrt((df[FEATS] ** 2).sum(axis=1))
    else:
        df["overall_error"] = df[FEATS].mean(axis=1)

    job[cache_key] = df.to_json(orient="split")
    return df


@csrf_exempt
@require_POST
def api_train(request):
    import json
    params = json.loads(request.body)
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {
        "status": "training", "loss": [],
        "epochs": int(params.get("epochs", 30)), "error": None,
        "df_err": None, "z_vals": None, "z_labs": None,
        "umap_cache": None, "umap_params": None,
    }

    def _worker():
        try:
            pipe = Pipeline(
                model_type=params["model_type"],
                seq_len=int(params["seq_len"]),
                ewma=float(params["ewma"]),
                epochs=int(params["epochs"]),
                batch_size=int(params["batch_size"]),
                hidden=int(params["hidden"]),
                layers=int(params["layers"]),
                lr=float(params["lr"]),
            )
            df_normal = DF[DF["run_id"].isin(NORMAL_RUNS)]
            pipe.train(df_normal)
            JOBS[job_id]["loss"] = pipe.loss_history

            df_err, z_vals, z_labs = pipe.evaluate(DF)
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(z_vals)
            JOBS[job_id].update({
                "status": "done",
                "df_err": df_err.to_json(orient="split"),
                "z_vals": z_vals.tolist(),
                "z_labs": z_labs,
                "umap_cache": emb.tolist(),
                "umap_params": "pca",
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
        "status": job["status"], "loss": job["loss"],
        "epochs": job["epochs"], "error": job.get("error"),
    })


def _get_job(request):
    job_id = request.GET.get("job_id")
    if job_id and job_id in JOBS and JOBS[job_id]["status"] == "done":
        return JOBS[job_id]
    for jid in reversed(list(JOBS)):
        if JOBS[jid]["status"] == "done":
            return JOBS[jid]
    return None


@require_GET
def api_chart_umap(request):
    job = _get_job(request)
    if not job:
        return JsonResponse({"error": "no completed job"}, status=400)

    method = request.GET.get("method", "pca")
    nn     = int(request.GET.get("nn", 15))
    md     = float(request.GET.get("md", 0.1))
    cb     = request.GET.get("color_by", "run")
    cache_key = "pca" if method == "pca" else f"umap_{nn}_{md}"

    z_vals = np.array(job["z_vals"])
    z_labs = job["z_labs"]

    if job.get("umap_params") != cache_key:
        if method == "pca":
            emb = PCA(n_components=2, random_state=42).fit_transform(z_vals)
        else:
            emb = umap_lib.UMAP(n_neighbors=nn, min_dist=md, random_state=42).fit_transform(z_vals)
        job["umap_cache"]  = emb.tolist()
        job["umap_params"] = cache_key
    else:
        emb = np.array(job["umap_cache"])

    fig = go.Figure()
    if cb == "run":
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

    title = "PCA Projection (Fast)" if method == "pca" else "UMAP Projection (Detailed)"
    fig.update_layout(**_LAYOUT, height=460,
        title=dict(text=title, font=dict(size=13, color="#0f172a")))
    return JsonResponse(_to_json(fig))


@require_GET
def api_chart_raw(request):
    run_id      = request.GET.get("run", ALL_RUNS[0])
    show_labels = request.GET.get("show_labels", "1") == "1"
    dfr         = DF[DF["run_id"] == run_id].reset_index(drop=True)
    fig         = go.Figure()
    for feat, clr in zip(FEATS, FEAT_COLORS):
        fig.add_trace(go.Scatter(
            x=dfr.index.tolist(), y=dfr[feat].tolist(),
            mode="lines", name=feat, line=dict(color=clr, width=1.5),
        ))
    if show_labels:
        fig = _add_anomaly_marks(fig, dfr, show_raw=True)
    fig.update_layout(**_LAYOUT, height=230,
        title=dict(text=f"Raw Sensor — {run_id}", font=dict(size=12, color="#0f172a")))
    return JsonResponse(_to_json(fig))


@require_GET
def api_chart_error(request):
    job         = _get_job(request)
    run_id      = request.GET.get("run", ANOMALY_RUNS[0])
    agg         = request.GET.get("agg", "mean")
    show_labels = request.GET.get("show_labels", "1") == "1"
    if not job:
        return JsonResponse({"error": "no completed job"}, status=400)

    df_err = _recompute_err(job, agg)
    dfr    = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
    df_raw = DF[DF["run_id"] == run_id].reset_index(drop=True)

    fig = go.Figure()
    for feat, clr in zip(FEATS, FEAT_COLORS):
        if feat in dfr.columns:
            fig.add_trace(go.Scatter(
                x=dfr.index.tolist(), y=dfr[feat].tolist(),
                mode="lines", name=feat, line=dict(color=clr, width=1, dash="dot"), opacity=0.6,
            ))
    fig.add_trace(go.Scatter(
        x=dfr.index.tolist(), y=dfr["overall_error"].tolist(),
        mode="lines", name="Overall", line=dict(color="#0f172a", width=2.5),
    ))
    if show_labels:
        fig = _add_anomaly_marks(fig, df_raw, show_raw=False)
    fig.update_layout(**_LAYOUT, height=230,
        title=dict(text=f"Reconstruction Error ({agg.upper()})", font=dict(size=12, color="#0f172a")))
    return JsonResponse(_to_json(fig))


@require_GET
def api_chart_inspector_umap(request):
    job    = _get_job(request)
    run_id = request.GET.get("run", ANOMALY_RUNS[0])
    if not job or not job.get("umap_cache"):
        return JsonResponse({"error": "no completed job"}, status=400)

    z_labs = job["z_labs"]
    emb    = np.array(job["umap_cache"])
    fig    = go.Figure()
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
        if not len(idx): continue
        fig.add_trace(go.Scatter(
            x=emb[idx, 0].tolist(), y=emb[idx, 1].tolist(),
            mode="markers", name=grp.replace("_", " "),
            marker=dict(size=sz, color=clr, opacity=op),
            customdata=[z_labs[i] for i in idx],
            hovertemplate="%{customdata}<extra></extra>",
        ))

    method_name = "PCA" if job.get("umap_params") == "pca" else "UMAP"
    fig.update_layout(**_LAYOUT, height=300,
        title=dict(text=f"{method_name} — {run_id} highlighted", font=dict(size=12, color="#0f172a")))
    return JsonResponse(_to_json(fig))


@require_GET
def api_chart_threshold(request):
    job         = _get_job(request)
    run_id      = request.GET.get("run", ANOMALY_RUNS[0])
    agg         = request.GET.get("agg", "mean")
    show_labels = request.GET.get("show_labels", "1") == "1"
    if not job:
        return JsonResponse({"error": "no completed job"}, status=400)

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
    df_err    = _recompute_err(job, agg)
    evaluator = ThresholdEvaluator(df_err[df_err["run_id"].isin(NORMAL_RUNS)]["overall_error"].dropna().values)
    dfr       = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
    df_raw    = DF[DF["run_id"] == run_id].reset_index(drop=True)
    th1, th2, th3, th4 = evaluator.calculate(dfr["overall_error"], cfg)
    err_vals  = dfr["overall_error"].fillna(0).values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfr.index.tolist(), y=dfr["overall_error"].tolist(),
        mode="lines", name="Overall Error",
        line=dict(color="#0f172a", width=2.5),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.03)",
    ))
    for name, vals, dash in [
        ("P99 Static", th1, "dot"), ("Sliding Mu+αStd", th2, "dash"),
        ("Adaptive-z", th3, "longdash"), ("Entropy-lock", th4, "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=dfr.index.tolist(), y=vals.tolist(),
            mode="lines", name=name,
            line=dict(color=TH_COLORS[name], dash=dash, width=1.8),
        ))

    # Per-threshold detection bands — Scatter fill="toself" per segment so toggle works.
    # First segment gets showlegend=True (the legend item), rest share legendgroup → hidden.
    _TH_BAND = {
        "P99 Static":      ("rgba(234,179,8,0.18)",    "rgba(234,179,8,0.65)",   "#ca8a04"),
        "Sliding Mu+αStd": ("rgba(59,130,246,0.13)",  "rgba(59,130,246,0.55)",  "#3b82f6"),
        "Adaptive-z":      ("rgba(16,185,129,0.13)",  "rgba(16,185,129,0.55)",  "#10b981"),
        "Entropy-lock":    ("rgba(124,58,237,0.13)",  "rgba(124,58,237,0.55)",  "#7c3aed"),
    }
    y_max = float(dfr["overall_error"].max()) * 1.15 or 1.0
    for th_name, th_vals in [
        ("P99 Static", th1), ("Sliding Mu+αStd", th2),
        ("Adaptive-z", th3), ("Entropy-lock", th4),
    ]:
        fill_rgba, marker_rgba, line_hex = _TH_BAND[th_name]
        anom = np.where(err_vals > th_vals)[0]
        band_group = f"band_{th_name}"
        if not len(anom):
            # Still add a dummy legend entry so the item always appears
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                name=f"{th_name} band",
                legendgroup=band_group,
                showlegend=True,
                marker=dict(size=10, color=marker_rgba, symbol="square",
                            line=dict(width=1, color=line_hex)),
                hoverinfo="skip",
            ))
            continue
        segs = np.split(anom, np.where(np.diff(anom) != 1)[0] + 1)
        for i, g in enumerate(segs):
            if not len(g):
                continue
            x0, x1 = int(g[0]) - 0.5, int(g[-1]) + 0.5
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[0, 0, y_max, y_max, 0],
                mode="lines", fill="toself",
                fillcolor=fill_rgba,
                line=dict(width=0),
                name=f"{th_name} band",
                legendgroup=band_group,
                showlegend=(i == 0),   # only first segment shows in legend
                legendrank=1000,
                marker=dict(color=marker_rgba, symbol="square",
                            line=dict(width=1, color=line_hex)),
                hoverinfo="skip",
            ))

    if show_labels:
        fig = _add_anomaly_marks(fig, df_raw, show_raw=False)

    height = int(request.GET.get("height", 400))
    fig.update_layout(**_LAYOUT, height=height,
        title=dict(text=f"Thresholds — {run_id} ({agg.upper()})", font=dict(size=12, color="#0f172a")))

    stats = {}
    metrics = {}  # per-threshold confusion matrix for THIS run

    for th_name, th_vals in [
        ("P99 Static", th1), ("Sliding Mu+αStd", th2),
        ("Adaptive-z", th3), ("Entropy-lock", th4),
    ]:
        n_flag = int((err_vals > th_vals).sum())
        stats[th_name] = {"flagged": n_flag, "pct": round(100 * n_flag / max(len(err_vals), 1), 1)}

        # ── Point-Adjusted confusion matrix for this run ──────────
        from ._shared import LABEL_COLS as _LC
        from ..ml import _point_adjust, _get_anomaly_segments

        has_labels = any(c in df_raw.columns for c in _LC)
        if has_labels:
            present = [c for c in _LC if c in df_raw.columns]
            y_true_raw = (df_raw[present].fillna(0).max(axis=1).values > 0).astype(int)
            n = len(err_vals)
            if len(y_true_raw) >= n:
                y_true = y_true_raw[:n]
            else:
                import numpy as _np
                y_true = _np.concatenate([y_true_raw, _np.zeros(n - len(y_true_raw), dtype=int)])

            valid_mask = ~dfr["overall_error"].isna().values
            y_pred = (err_vals > th_vals).astype(int)
            y_adj  = _point_adjust(y_pred, y_true)

            yp = y_adj[valid_mask]
            yt = y_true[valid_mask]

            tp = int(((yp == 1) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            tn = int(((yp == 0) & (yt == 0)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            acc  = (tp + tn) / max(tp + fp + tn + fn, 1)

            segs      = _get_anomaly_segments(y_true)
            n_segs    = len(segs)
            detected  = sum(1 for s, e in segs if y_pred[s:e+1].any())

            metrics[th_name] = dict(
                tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(prec, 4), recall=round(rec, 4),
                f1=round(f1, 4), accuracy=round(acc, 4),
                n_segments=n_segs, n_detected_segs=detected,
                has_labels=True,
            )
        else:
            metrics[th_name] = {"has_labels": False}

    result = _to_json(fig)
    result["stats"]   = stats
    result["metrics"] = metrics
    return JsonResponse(result)
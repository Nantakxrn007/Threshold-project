# dashboard/views/pr_curve.py
"""
PR Plot API: compute Point-Adjusted Precision & Recall for each
TH1–TH4 operating point across ALL models, then render as a
scatter plot (color = model, symbol = threshold).
No threshold sweep — just the 4 operating points per model.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ._shared import (
    DF, NORMAL_RUNS, ANOMALY_RUNS, FEATS, LABEL_COLS,
    TH_NAMES, _LAYOUT, _to_json, _cfg_from_request,
    ThresholdEvaluator, MODEL_CHOICES,
)
from ..ml import _point_adjust
from .batch import _get_batch_job


# ── Visual constants ──────────────────────────────────────────────────────────
# One color per model slot (up to 8 models)
_MODEL_COLORS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#db2777",  # pink
    "#65a30d",  # lime
]

# One Plotly marker symbol per threshold method
_TH_SYMBOLS = {
    "P99 Static":      "circle",
    "Sliding Mu+αStd": "diamond",
    "Adaptive-z":      "square",
    "Entropy-lock":    "star",
}

# Nice short labels for legend / hover
_TH_SHORT = {
    "P99 Static":      "P99",
    "Sliding Mu+αStd": "Sliding",
    "Adaptive-z":      "Adaptive",
    "Entropy-lock":    "Entropy",
}


def _pa_metrics(y_pred: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray):
    """Point-Adjusted precision, recall, f1 for one threshold array."""
    y_adj = _point_adjust(y_pred, y_true)
    yp = y_adj[valid_mask]
    yt = y_true[valid_mask]
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


def _collect_errors(df_err: pd.DataFrame, agg: str):
    """
    For each anomaly run, collect (error_vals, y_true, valid_mask).
    Returns concatenated arrays.
    """
    all_err, all_true, all_valid = [], [], []
    for run_id in ANOMALY_RUNS:
        dfr = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
        if dfr.empty:
            continue
        df_run = DF[DF["run_id"] == run_id].reset_index(drop=True)
        present = [c for c in LABEL_COLS if c in df_run.columns]
        if not present:
            continue

        err_vals   = dfr["overall_error"].fillna(0).values
        valid_mask = ~dfr["overall_error"].isna().values
        y_true_raw = (df_run[present].fillna(0).max(axis=1).values > 0).astype(int)
        n = len(err_vals)
        y_true = (y_true_raw[:n] if len(y_true_raw) >= n
                  else np.concatenate([y_true_raw, np.zeros(n - len(y_true_raw), dtype=int)]))

        all_err.append(err_vals)
        all_true.append(y_true)
        all_valid.append(valid_mask)

    if not all_err:
        return None, None, None
    return np.concatenate(all_err), np.concatenate(all_true), np.concatenate(all_valid)


def _apply_agg(df_err: pd.DataFrame, agg: str) -> pd.DataFrame:
    df = df_err.copy()
    if agg == "max":
        df["overall_error"] = df[FEATS].max(axis=1)
    elif agg == "l2":
        df["overall_error"] = np.sqrt((df[FEATS] ** 2).sum(axis=1))
    else:  # mean
        df["overall_error"] = df[FEATS].mean(axis=1)
    return df


def _operating_points(df_err: pd.DataFrame, err_cat, true_cat, valid_cat, cfg) -> dict:
    """Compute TH1–TH4 (precision, recall, f1) operating points for one model."""
    normal_errors = df_err[df_err["run_id"].isin(NORMAL_RUNS)]["overall_error"].dropna().values
    evaluator = ThresholdEvaluator(normal_errors)

    th_medians = {}  # th_name → list of median threshold values per run
    for run_id in ANOMALY_RUNS:
        dfr = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
        if dfr.empty:
            continue
        th_arrays = evaluator.calculate(dfr["overall_error"], cfg)
        for th_name, th_vals in zip(TH_NAMES, th_arrays):
            th_medians.setdefault(th_name, []).append(float(np.nanmedian(th_vals)))

    result = {}
    for th_name, vals in th_medians.items():
        t_op = float(np.mean(vals))
        y_pred_op = (err_cat > t_op).astype(int)
        p, r, f1 = _pa_metrics(y_pred_op, true_cat, valid_cat)
        result[th_name] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "threshold": round(t_op, 5),
        }
    return result


def _f1_isoline(f1_val: float, n: int = 120):
    """Points for an F1 iso-curve: precision = F1·recall / (2·recall − F1)."""
    r_min = f1_val / 2.0 + 0.001
    r_arr = np.linspace(r_min, 1.0, n)
    p_arr = np.clip(f1_val * r_arr / (2 * r_arr - f1_val), 0, 1)
    return r_arr.tolist(), p_arr.tolist()


@require_GET
def api_pr_curve(request):
    job = _get_batch_job(request)
    agg = request.GET.get("agg", "max")
    cfg = _cfg_from_request(request)

    if not job:
        return JsonResponse({"error": "no completed batch job"}, status=400)

    available_models = [m for m in MODEL_CHOICES if m in job["results"]]
    if not available_models:
        return JsonResponse({"error": "no results"}, status=400)

    # ── Build per-model operating points ─────────────────────────────
    model_points = {}   # model_name → { th_name → {p, r, f1, threshold} }

    for model_type in available_models:
        df_err = _apply_agg(
            pd.read_json(job["results"][model_type]["df_err"], orient="split"),
            agg,
        )
        err_cat, true_cat, valid_cat = _collect_errors(df_err, agg)
        if err_cat is None:
            continue
        model_points[model_type] = _operating_points(df_err, err_cat, true_cat, valid_cat, cfg)

    if not model_points:
        return JsonResponse({"error": "no labeled anomaly runs"}, status=400)

    # ── Build Plotly figure ───────────────────────────────────────────
    fig = go.Figure()

    # F1 iso-lines (faint background guide curves)
    for f1_val, iso_label in [(0.2, "0.2"), (0.4, "0.4"), (0.6, "0.6"), (0.8, "0.8")]:
        rx, py = _f1_isoline(f1_val)
        fig.add_trace(go.Scatter(
            x=rx, y=py,
            mode="lines",
            name=f"F1={iso_label}",
            line=dict(color="#cbd5e1", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Label near the top-left end of each curve
        label_idx = min(10, len(rx) - 1)
        fig.add_annotation(
            x=rx[label_idx], y=py[label_idx],
            text=f"F1={iso_label}",
            showarrow=False,
            font=dict(size=9, color="#94a3b8"),
            xanchor="center",
            yanchor="bottom",
        )

    # "Best Zone" shaded corner annotation
    fig.add_shape(
        type="rect",
        x0=0.75, y0=0.75, x1=1.0, y1=1.0,
        fillcolor="rgba(34,197,94,0.06)",
        line=dict(color="rgba(34,197,94,0.25)", width=1, dash="dot"),
        layer="below",
    )
    fig.add_annotation(
        x=0.875, y=0.78,
        text="✦ Best Zone",
        showarrow=False,
        font=dict(size=9, color="rgba(22,163,74,0.7)"),
        xanchor="center",
    )

    # One scatter group per model (4 threshold points each)
    for model_idx, (model_name, th_dict) in enumerate(model_points.items()):
        color = _MODEL_COLORS[model_idx % len(_MODEL_COLORS)]
        # Models beyond the first 2 are hidden by default in the legend
        visible = True if model_idx < 2 else "legendonly"

        for th_name in TH_NAMES:
            pt = th_dict.get(th_name)
            if pt is None:
                continue

            short = _TH_SHORT.get(th_name, th_name)
            symbol = _TH_SYMBOLS.get(th_name, "circle")
            f1pct  = f"{pt['f1'] * 100:.1f}%"

            fig.add_trace(go.Scatter(
                x=[pt["recall"]],
                y=[pt["precision"]],
                mode="markers+text",
                name=f"{model_name} · {short}",
                legendgroup=model_name,
                legendgrouptitle=dict(text=model_name) if th_name == TH_NAMES[0] else None,
                showlegend=True,
                visible=visible,
                marker=dict(
                    size=16,
                    color=color,
                    symbol=symbol,
                    opacity=0.92,
                    line=dict(width=2, color="white"),
                ),
                text=[short],
                textposition="top center",
                textfont=dict(size=9, color=color, family="JetBrains Mono, monospace"),
                hovertemplate=(
                    f"<b>{model_name}</b> · {th_name}<br>"
                    "Recall = %{x:.3f}<br>"
                    "Precision = %{y:.3f}<br>"
                    f"F1 = {f1pct}<extra></extra>"
                ),
                customdata=[[pt["f1"], pt["threshold"]]],
            ))

    # ── Layout ────────────────────────────────────────────────────────
    layout_base = {k: v for k, v in _LAYOUT.items()
                   if k not in ("xaxis", "yaxis", "legend")}
    fig.update_layout(
        **layout_base,
        height=460,
        xaxis=dict(
            title="Recall",
            range=[-0.03, 1.06],
            gridcolor="#e2e8f0",
            linecolor="#e2e8f0",
            color="#94a3b8",
            zeroline=False,
            tickformat=".0%",
        ),
        yaxis=dict(
            title="Precision",
            range=[-0.03, 1.06],
            gridcolor="#e2e8f0",
            linecolor="#e2e8f0",
            color="#94a3b8",
            zeroline=False,
            tickformat=".0%",
        ),
        title=dict(
            text=f"PR Plot — Precision × Recall ({agg.upper()})&nbsp;"
                 f"<span style='font-size:11px;color:#94a3b8;'>"
                 f"คลิก legend เพื่อเลือก/ซ่อนโมเดล</span>",
            font=dict(size=13, color="#0f172a"),
        ),
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            tracegroupgap=6,
            font=dict(size=11),
        ),
    )

    # ── Symbol legend annotation (bottom of chart) ────────────────────
    symbol_hint = " &nbsp;·&nbsp; ".join(
        f"<b>{sym.title()}</b> = {_TH_SHORT[th]}"
        for th, sym in _TH_SYMBOLS.items()
    )
    fig.add_annotation(
        text=symbol_hint,
        xref="paper", yref="paper",
        x=0, y=-0.10,
        xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=9, color="#94a3b8"),
    )

    result = _to_json(fig)
    result["models_available"] = list(model_points.keys())
    result["model_points"]     = {
        m: {th: pt for th, pt in thd.items()}
        for m, thd in model_points.items()
    }
    return JsonResponse(result)
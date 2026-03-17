# dashboard/views/_shared.py
"""Shared imports, constants, and helpers for all view modules."""
import json
import threading
import uuid

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import umap.umap_ as umap_lib
from sklearn.decomposition import PCA
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from ..ml import (
    ALL_RUNS, ANOMALY_RUNS, FEATS, NORMAL_RUNS, MODEL_MAP, LABEL_COLS,
    Pipeline, ThresholdEvaluator, get_training_data, compute_metrics_from_error,
)

# ── Globals ───────────────────────────────────────────────────────────────────
DF = get_training_data()

JOBS: dict       = {}
BATCH_JOBS: dict = {}

MODEL_CHOICES = [
    "LSTM-AE", "GRU-AE", "CNN-LSTM-AE",
    "Plain-AE", "Plain-LSTM", "LSTM-Attention-AE",
]

# ── Plotly theme ──────────────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, system-ui, sans-serif", color="#64748b", size=11),
    xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", color="#94a3b8", zeroline=False),
    yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", color="#94a3b8", zeroline=False),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                    font=dict(color="#0f172a", family="Inter, system-ui, sans-serif")),
    margin=dict(l=50, r=16, t=44, b=36),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=10)),
)

FEAT_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899"]

TH_COLORS = {
    "P99 Static":      "#ef4444",
    "Sliding Mu+αStd": "#3b82f6",
    "Adaptive-z":      "#10b981",
    "Entropy-lock":    "#7c3aed",
}

TH_NAMES = ["P99 Static", "Sliding Mu+αStd", "Adaptive-z", "Entropy-lock"]


def _to_json(fig: go.Figure) -> dict:
    return json.loads(plotly.io.to_json(fig))


def _cfg_from_request(request) -> dict:
    """Parse threshold config from GET params."""
    g = request.GET.get
    return {
        "th1_pct":    float(g("th1_pct",    99.0)),
        "th1_mode":         g("th1_mode",   "sliding"),
        "th1_win":    float(g("th1_win",   100)),
        "th1_recalc": float(g("th1_recalc",  10)),
        "th2_alpha":  float(g("th2_alpha",   3.5)),
        "th2_win":    float(g("th2_win",    80)),
        "th2_recalc": float(g("th2_recalc", 50)),
        "th3_zmin":   float(g("th3_zmin",   2.0)),
        "th3_zmax":   float(g("th3_zmax",  10.0)),
        "th3_win":    float(g("th3_win",   80)),
        "th3_recalc": float(g("th3_recalc", 1)),
        "th4_alpha":  float(g("th4_alpha",  3.5)),
        "th4_win":    float(g("th4_win",  150)),
        "th4_cons":   float(g("th4_cons",   5)),
        "th4_eth":    float(g("th4_eth",  0.95)),
        "th4_recalc": float(g("th4_recalc", 1)),
    }



def _add_anomaly_marks(fig: go.Figure, df_run: pd.DataFrame,
                       show_raw: bool = False) -> go.Figure:
    """
    Add anomaly label overlays to a figure.

    show_raw=True  (raw sensor chart):
        Per-sensor scatter markers plotted at actual sensor y-values
        where that sensor's label == 1. Each sensor is a separate toggle.

    show_raw=False (error / threshold chart):
        Single "Anomaly Label" band using vrect for each anomaly segment
        (any sensor). Toggleable via a dummy scatter legend item.
    """
    # Check which label columns are actually present
    present_labels = [c for c in LABEL_COLS if c in df_run.columns]
    if not present_labels:
        return fig

    n = len(df_run)
    idx = list(range(n))

    if show_raw:
        # ── Per-sensor markers on top of sensor lines ──────────────
        # Map: sensor feature → (label_col, line_color, marker_color)
        sensor_cfg = [
            ("conductivity", "Anomaly C_filled", "#3b82f6", "#1d4ed8"),
            ("pH",           "Anomaly P_filled", "#10b981", "#047857"),
            ("temperature",  "Anomaly T_filled", "#f59e0b", "#b45309"),
            ("voltage",      "Anomaly V_filled", "#ec4899", "#be185d"),
        ]
        any_added = False
        for feat, label_col, line_clr, marker_clr in sensor_cfg:
            if label_col not in df_run.columns:
                continue
            if feat not in df_run.columns:
                continue
            mask = (df_run[label_col].fillna(0) > 0).values
            if not mask.any():
                continue  # This sensor has no anomaly in this run

            anom_idx = [i for i in idx if mask[i]]
            anom_y   = [float(df_run[feat].iloc[i]) for i in anom_idx]

            fig.add_trace(go.Scatter(
                x=anom_idx,
                y=anom_y,
                mode="markers",
                name=f"⚠ Label: {feat}",
                marker=dict(
                    color=marker_clr,
                    size=6,
                    symbol="circle-open",
                    line=dict(width=2, color=marker_clr),
                ),
                legendgroup="anomaly_marks",
                legendgrouptitle=dict(text="Anomaly Labels") if not any_added else None,
                showlegend=True,
                hovertemplate=f"[{feat}] anomaly idx=%{{x}}<extra></extra>",
            ))
            any_added = True

    else:
        # ── Overall vrect bands (any sensor) ──────────────────────
        # Combine all present label columns
        present = [c for c in LABEL_COLS if c in df_run.columns]
        overall_mask = (df_run[present].fillna(0).max(axis=1) > 0).values

        # Build contiguous segments
        segs = []
        in_s = False
        for i, v in enumerate(overall_mask):
            if v and not in_s:
                s = i; in_s = True
            elif not v and in_s:
                segs.append((s, i - 1)); in_s = False
        if in_s:
            segs.append((s, n - 1))

        if not segs:
            return fig

        for seg_s, seg_e in segs:
            fig.add_vrect(
                x0=seg_s, x1=seg_e,
                fillcolor="rgba(220,38,38,0.13)",
                layer="below", line_width=0,
            )

        # Single dummy scatter for legend toggle
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color="rgba(220,38,38,0.5)", symbol="square"),
            name="⚠ Anomaly Label",
            legendgroup="anomaly_marks",
            showlegend=True,
            hoverinfo="skip",
        ))

    return fig
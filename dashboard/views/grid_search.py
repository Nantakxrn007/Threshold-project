# dashboard/views/grid_search.py
"""
Grid Search: sweep model params x threshold configs.
Scoring uses ANOMALY RUNS ONLY — normal runs used only for threshold calibration.

Search modes:
  grid   - exhaustive
  random - sample N random combinations
  optuna - Bayesian via Optuna (pip install optuna)
"""
import json, threading, uuid, traceback, random as _random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from ._shared import (
    DF, NORMAL_RUNS, ANOMALY_RUNS, ALL_RUNS, FEATS, LABEL_COLS,
    FEAT_COLORS, TH_COLORS, TH_NAMES, _LAYOUT, _to_json,
    _add_anomaly_marks, Pipeline, ThresholdEvaluator,
)
from ..ml import _point_adjust, _get_anomaly_segments

GRID_JOBS: dict = {}
GRID_STOP_FLAGS: dict = {}

_DEFAULT_CFG = dict(
    th1_pct=99.0, th1_mode="sliding", th1_win=100, th1_recalc=10,
    th2_alpha=3.5, th2_win=80, th2_recalc=50,
    th3_zmin=2.0, th3_zmax=10.0, th3_win=80, th3_recalc=1,
    th4_alpha=3.5, th4_win=150, th4_cons=5, th4_eth=0.95, th4_recalc=1,
)


def _pick_th(th_arrays, th_type):
    try:
        idx = TH_NAMES.index(th_type)
    except ValueError:
        idx = 0
    return th_arrays[idx]


def _eval_combo(df_err, cfg, th_type, agg):
    """
    Score on ANOMALY RUNS ONLY.
    Normal runs are intentionally excluded:
      - they have no true anomaly labels (y_true = all 0)
      - including them would give F1=0 for every normal run
      - this would drag down the mean and make all combos look equally bad
    Normal runs ARE used by ThresholdEvaluator to calibrate the threshold level.
    """
    df_e = df_err.copy()
    if agg == "max":
        df_e["overall_error"] = df_e[FEATS].max(axis=1)
    elif agg == "l2":
        df_e["overall_error"] = np.sqrt((df_e[FEATS] ** 2).sum(axis=1))
    else:
        df_e["overall_error"] = df_e[FEATS].mean(axis=1)

    normal_errors = df_e[df_e["run_id"].isin(NORMAL_RUNS)]["overall_error"].dropna().values
    evaluator = ThresholdEvaluator(normal_errors)

    has_labels = any(c in DF.columns for c in LABEL_COLS)
    all_f1, all_rec, all_prec, all_acc = [], [], [], []
    total_segs, total_det = 0, 0

    for run_id in ANOMALY_RUNS:
        dfr = df_e[df_e["run_id"] == run_id].reset_index(drop=True)
        if dfr.empty:
            continue
        th_arrays = evaluator.calculate(dfr["overall_error"], cfg)
        th_vals = _pick_th(th_arrays, th_type)
        err_vals = dfr["overall_error"].fillna(0).values
        valid_mask = ~dfr["overall_error"].isna().values
        y_pred = (err_vals > th_vals).astype(int)

        if not has_labels:
            continue

        df_run = DF[DF["run_id"] == run_id].reset_index(drop=True)
        present = [c for c in LABEL_COLS if c in df_run.columns]
        if not present:
            continue

        y_true_raw = (df_run[present].fillna(0).max(axis=1).values > 0).astype(int)
        n = len(err_vals)
        if len(y_true_raw) >= n:
            y_true = y_true_raw[:n]
        else:
            y_true = np.concatenate([y_true_raw, np.zeros(n - len(y_true_raw), dtype=int)])

        y_adj = _point_adjust(y_pred, y_true)
        yp = y_adj[valid_mask]
        yt = y_true[valid_mask]

        tp = int(((yp==1)&(yt==1)).sum())
        fp = int(((yp==1)&(yt==0)).sum())
        fn = int(((yp==0)&(yt==1)).sum())
        tn = int(((yp==0)&(yt==0)).sum())
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-9)
        acc  = (tp+tn) / max(tp+fp+tn+fn, 1)
        segs = _get_anomaly_segments(y_true)
        det  = sum(1 for s, e in segs if y_pred[s:e+1].any())

        all_f1.append(f1); all_rec.append(rec)
        all_prec.append(prec); all_acc.append(acc)
        total_segs += len(segs); total_det += det

    mean = lambda arr: float(np.mean(arr)) if arr else 0.0
    return dict(
        f1=round(mean(all_f1), 4),
        recall=round(mean(all_rec), 4),
        precision=round(mean(all_prec), 4),
        accuracy=round(mean(all_acc), 4),
        n_segments=total_segs,
        n_detected_segs=total_det,
    )


def _calc_score(metrics, scoring):
    if scoring == "f1_recall":
        return 0.6*metrics["f1"] + 0.4*metrics["recall"]
    elif scoring == "recall":
        return metrics["recall"]
    return metrics["f1"]


def _make_combo(cid, arch, h, seq, ep, lr, ewma, batch_size, layers, tc):
    return dict(
        id=cid, arch=arch, hidden=h, seq_len=seq, epochs=ep,
        lr=lr, ewma=ewma, batch_size=batch_size, layers=layers,
        th_name=tc["name"], th_type=tc["th_type"], th_cfg=tc["cfg"],
        epoch_progress=0, status="pending",
        f1=None, recall=None, precision=None,
        accuracy=None, score=None,
        n_segments=0, n_detected_segs=0,
    )


def _run_combo(job_id, combo, model_cache, agg, scoring):
    i = combo["id"]
    GRID_JOBS[job_id]["combos"][i]["status"] = "training"
    mkey = (combo["arch"], combo["hidden"], combo["seq_len"],
            combo["epochs"], combo["lr"], combo["ewma"])

    if mkey not in model_cache:
        df_normal = DF[DF["run_id"].isin(NORMAL_RUNS)]
        pipe = Pipeline(
            model_type=combo["arch"],
            seq_len=combo["seq_len"],
            ewma=combo["ewma"],
            epochs=combo["epochs"],
            batch_size=combo["batch_size"],
            hidden=combo["hidden"],
            layers=combo["layers"],
            lr=combo["lr"],
        )
        def _cb(ep, loss, _ep=combo["epochs"], _i=i):
            if _i < len(GRID_JOBS[job_id]["combos"]):
                GRID_JOBS[job_id]["combos"][_i]["epoch_progress"] = round(100*ep/max(_ep,1))
        pipe.train(df_normal, on_epoch=_cb)
        df_err, _, _ = pipe.evaluate(DF, agg=agg)
        model_cache[mkey] = df_err

    df_err = model_cache[mkey]
    GRID_JOBS[job_id]["combos"][i]["epoch_progress"] = 100
    metrics = _eval_combo(df_err, combo["th_cfg"], combo["th_type"], agg)
    score   = _calc_score(metrics, scoring)
    GRID_JOBS[job_id]["combos"][i].update({**metrics, "score": round(score, 4), "status": "done"})
    GRID_JOBS[job_id]["done"] = GRID_JOBS[job_id].get("done", 0) + 1
    return score, df_err, mkey


def _finalise(job_id, model_cache, agg):
    done = [c for c in GRID_JOBS[job_id]["combos"] if c["status"] == "done"]
    if not done:
        return
    best = max(done, key=lambda x: x["score"] or 0)
    GRID_JOBS[job_id]["best_idx"] = best["id"]
    mkey = (best["arch"], best["hidden"], best["seq_len"],
            best["epochs"], best["lr"], best["ewma"])
    df_b = model_cache.get(mkey)
    if df_b is None:
        return
    df_b = df_b.copy()
    if agg == "max":
        df_b["overall_error"] = df_b[FEATS].max(axis=1)
    elif agg == "l2":
        df_b["overall_error"] = np.sqrt((df_b[FEATS]**2).sum(axis=1))
    else:
        df_b["overall_error"] = df_b[FEATS].mean(axis=1)
    GRID_JOBS[job_id]["best_df_err_json"] = df_b.to_json(orient="split")
    GRID_JOBS[job_id]["best_th_cfg"]  = best["th_cfg"]
    GRID_JOBS[job_id]["best_th_type"] = best["th_type"]


def _worker_grid(job_id, agg, scoring):
    try:
        model_cache = {}
        for combo in list(GRID_JOBS[job_id]["combos"]):
            if GRID_STOP_FLAGS.get(job_id):
                GRID_JOBS[job_id]["status"] = "stopped"
                GRID_STOP_FLAGS.pop(job_id, None)
                _finalise(job_id, model_cache, agg)
                return
            _run_combo(job_id, combo, model_cache, agg, scoring)
        _finalise(job_id, model_cache, agg)
        GRID_JOBS[job_id]["status"] = "done"
    except Exception as exc:
        GRID_JOBS[job_id].update({"status": "error", "error": str(exc) + "\n" + traceback.format_exc()})


def _worker_random(job_id, param_space, n_trials, batch_size, layers, th_configs, agg, scoring):
    try:
        model_cache = {}
        cid = 0
        for _ in range(n_trials):
            if GRID_STOP_FLAGS.get(job_id):
                GRID_JOBS[job_id]["status"] = "stopped"
                GRID_STOP_FLAGS.pop(job_id, None)
                _finalise(job_id, model_cache, agg)
                return
            tc = _random.choice(th_configs)
            combo = _make_combo(
                cid,
                _random.choice(param_space["archs"]),
                _random.choice(param_space["hiddens"]),
                _random.choice(param_space["seq_lens"]),
                _random.choice(param_space["epochs_list"]),
                _random.choice(param_space["lrs"]),
                _random.choice(param_space["ewmas"]),
                batch_size, layers, tc,
            )
            GRID_JOBS[job_id]["combos"].append(combo)
            GRID_JOBS[job_id]["total"] = cid + 1
            cid += 1
            _run_combo(job_id, combo, model_cache, agg, scoring)
        _finalise(job_id, model_cache, agg)
        GRID_JOBS[job_id]["status"] = "done"
    except Exception as exc:
        GRID_JOBS[job_id].update({"status": "error", "error": str(exc) + "\n" + traceback.format_exc()})


def _worker_optuna(job_id, param_space, n_trials, batch_size, layers, th_configs, agg, scoring):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        model_cache = {}
        cid = [0]

        def objective(trial):
            if GRID_STOP_FLAGS.get(job_id):
                raise optuna.exceptions.OptunaError("stopped")
            tc_i  = trial.suggest_categorical("th_idx", list(range(len(th_configs))))
            tc    = th_configs[tc_i]
            combo = _make_combo(
                cid[0],
                trial.suggest_categorical("arch",   param_space["archs"]),
                trial.suggest_categorical("hidden", param_space["hiddens"]),
                trial.suggest_categorical("seq",    param_space["seq_lens"]),
                trial.suggest_categorical("epochs", param_space["epochs_list"]),
                trial.suggest_categorical("lr",     param_space["lrs"]),
                trial.suggest_categorical("ewma",   param_space["ewmas"]),
                batch_size, layers, tc,
            )
            GRID_JOBS[job_id]["combos"].append(combo)
            GRID_JOBS[job_id]["total"] = cid[0] + 1
            cid[0] += 1
            score, _, _ = _run_combo(job_id, combo, model_cache, agg, scoring)
            return score

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except Exception:
            pass
        _finalise(job_id, model_cache, agg)
        stopped = GRID_STOP_FLAGS.pop(job_id, False)
        GRID_JOBS[job_id]["status"] = "stopped" if stopped else "done"
    except ImportError:
        GRID_JOBS[job_id].update({
            "status": "error",
            "error": "Optuna ไม่ได้ติดตั้ง กรุณารัน: pip install optuna แล้วลองใหม่",
        })
    except Exception as exc:
        GRID_JOBS[job_id].update({"status": "error", "error": str(exc) + "\n" + traceback.format_exc()})


@csrf_exempt
@require_POST
def api_grid_search_start(request):
    p = json.loads(request.body)

    archs       = p.get("archs",       ["LSTM-AE"])
    hiddens     = [int(x)   for x in p.get("hiddens",      [16])]
    seq_lens    = [int(x)   for x in p.get("seq_lens",     [10])]
    epochs_list = [int(x)   for x in p.get("epochs_list",  [30])]
    lrs         = [float(x) for x in p.get("lrs",        [0.005])]
    ewmas       = [float(x) for x in p.get("ewmas",       [0.3])]
    batch_size  = int(p.get("batch_size", 64))
    layers      = int(p.get("layers", 1))
    agg         = p.get("agg",         "max")
    scoring     = p.get("scoring",     "mean_f1")
    search_mode = p.get("search_mode", "grid")
    n_trials    = max(int(p.get("n_trials", 30)), 1)
    th_configs  = p.get("th_configs", [])

    if not th_configs:
        th_configs = [{"name": "P99 Static", "th_type": "P99 Static", "cfg": _DEFAULT_CFG}]

    param_space = dict(archs=archs, hiddens=hiddens, seq_lens=seq_lens,
                       epochs_list=epochs_list, lrs=lrs, ewmas=ewmas)

    job_id = uuid.uuid4().hex[:8]

    if search_mode == "grid":
        all_combos, cid = [], 0
        for arch in archs:
            for h in hiddens:
                for seq in seq_lens:
                    for ep in epochs_list:
                        for lr in lrs:
                            for ewma in ewmas:
                                for tc in th_configs:
                                    all_combos.append(_make_combo(
                                        cid, arch, h, seq, ep, lr, ewma, batch_size, layers, tc))
                                    cid += 1
        total = len(all_combos)
    else:
        all_combos = []
        total = n_trials

    GRID_JOBS[job_id] = {
        "status":  "running", "mode": search_mode,
        "total":   total,     "done": 0,
        "agg":     agg,       "scoring": scoring,
        "combos":  all_combos,
        "best_idx": None, "best_df_err_json": None,
        "best_th_cfg": None, "best_th_type": None,
        "error": None,
    }

    if search_mode == "optuna":
        t = threading.Thread(target=_worker_optuna,
            args=(job_id, param_space, n_trials, batch_size, layers, th_configs, agg, scoring),
            daemon=True)
    elif search_mode == "random":
        t = threading.Thread(target=_worker_random,
            args=(job_id, param_space, n_trials, batch_size, layers, th_configs, agg, scoring),
            daemon=True)
    else:
        t = threading.Thread(target=_worker_grid, args=(job_id, agg, scoring), daemon=True)

    t.start()
    return JsonResponse({"job_id": job_id, "total": total, "mode": search_mode})


@require_GET
def api_grid_search_status(request, job_id):
    job = GRID_JOBS.get(job_id)
    if not job:
        return JsonResponse({"error": "not found"}, status=404)
    combos_light = [{k: v for k, v in c.items() if k != "th_cfg"} for c in job["combos"]]
    return JsonResponse({
        "status": job["status"], "mode": job.get("mode", "grid"),
        "total":  job["total"],  "done": job["done"],
        "best_idx": job["best_idx"],
        "combos":   combos_light,
        "error":    job.get("error"),
    })


@require_GET
def api_grid_search_best_charts(request, job_id):
    job = GRID_JOBS.get(job_id)
    if not job or job["status"] != "done":
        return JsonResponse({"error": "not ready"}, status=400)
    if job["best_idx"] is None or job["best_df_err_json"] is None:
        return JsonResponse({"error": "no best result"}, status=400)

    best        = job["combos"][job["best_idx"]]
    run_id      = request.GET.get("run", ANOMALY_RUNS[0])
    show_labels = request.GET.get("show_labels", "1") == "1"
    agg         = job["agg"]
    cfg         = job["best_th_cfg"]
    th_type     = job["best_th_type"]

    df_err = pd.read_json(job["best_df_err_json"], orient="split")
    df_raw = DF[DF["run_id"] == run_id].reset_index(drop=True)
    dfr    = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
    if dfr.empty:
        return JsonResponse({"error": f"run {run_id} not in results"}, status=400)

    fig_raw = go.Figure()
    for feat, clr in zip(FEATS, FEAT_COLORS):
        if feat in df_raw.columns:
            fig_raw.add_trace(go.Scatter(
                x=df_raw.index.tolist(), y=df_raw[feat].tolist(),
                mode="lines", name=feat, line=dict(color=clr, width=1.5),
            ))
    if show_labels:
        fig_raw = _add_anomaly_marks(fig_raw, df_raw, show_raw=True)
    fig_raw.update_layout(**_LAYOUT, height=220,
        title=dict(text=f"Raw Sensor — {run_id}", font=dict(size=12, color="#0f172a")))

    normal_errors = df_err[df_err["run_id"].isin(NORMAL_RUNS)]["overall_error"].dropna().values
    evaluator     = ThresholdEvaluator(normal_errors)
    th1, th2, th3, th4 = evaluator.calculate(dfr["overall_error"], cfg)
    best_th  = _pick_th((th1, th2, th3, th4), th_type)
    err_vals = dfr["overall_error"].fillna(0).values

    fig_th = go.Figure()
    fig_th.add_trace(go.Scatter(
        x=dfr.index.tolist(), y=dfr["overall_error"].tolist(),
        mode="lines", name=f"Error ({agg.upper()})",
        line=dict(color="#0f172a", width=2.5),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.03)",
    ))
    for tname, tvals, dash in [
        ("P99 Static", th1, "dot"), ("Sliding Mu+αStd", th2, "dash"),
        ("Adaptive-z", th3, "longdash"), ("Entropy-lock", th4, "solid"),
    ]:
        is_best = (tname == th_type)
        fig_th.add_trace(go.Scatter(
            x=dfr.index.tolist(), y=tvals.tolist(), mode="lines",
            name=tname + (" *" if is_best else ""),
            line=dict(color=TH_COLORS[tname], dash=dash, width=2.8 if is_best else 1.4),
        ))
    _TH_BAND = {
        "P99 Static":      "rgba(239,68,68,0.10)",
        "Sliding Mu+αStd": "rgba(59,130,246,0.10)",
        "Adaptive-z":      "rgba(16,185,129,0.10)",
        "Entropy-lock":    "rgba(124,58,237,0.10)",
    }
    for _thn, _thv in [("P99 Static",th1),("Sliding Mu+αStd",th2),
                        ("Adaptive-z",th3),("Entropy-lock",th4)]:
        _anom = np.where(err_vals > _thv)[0]
        if not len(_anom): continue
        for _g in np.split(_anom, np.where(np.diff(_anom)!=1)[0]+1):
            if len(_g):
                fig_th.add_vrect(x0=int(_g[0])-0.5, x1=int(_g[-1])+0.5,
                    fillcolor=_TH_BAND.get(_thn,"rgba(239,68,68,0.07)"),
                    layer="below", line_width=0)
    if show_labels:
        fig_th = _add_anomaly_marks(fig_th, df_raw, show_raw=False)
    fig_th.update_layout(**_LAYOUT, height=320,
        title=dict(text=f"Best: {best['arch']} + {th_type} — {run_id} ({agg.upper()})",
                   font=dict(size=12, color="#0f172a")))

    best_info = {k: v for k, v in best.items() if k != "th_cfg"}
    return JsonResponse({"raw": _to_json(fig_raw), "threshold": _to_json(fig_th), "best": best_info})


@csrf_exempt
@require_POST
def api_grid_search_stop(request):
    """Stop grid search at current combo, keep partial results."""
    p      = json.loads(request.body) if request.body else {}
    job_id = p.get("job_id", "")
    if job_id and job_id in GRID_JOBS:
        GRID_STOP_FLAGS[job_id] = True
        GRID_JOBS[job_id]["status"] = "stopping"
    return JsonResponse({"ok": True})


@require_GET
def api_grid_search_export(request, job_id):
    """Export partial or full grid search results as CSV."""
    import io, csv
    job = GRID_JOBS.get(job_id)
    if not job:
        return JsonResponse({"error": "not found"}, status=404)

    done = [c for c in job["combos"] if c["status"] == "done"]
    done.sort(key=lambda x: x.get("score") or 0, reverse=True)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "rank","arch","hidden","seq_len","epochs","lr","ewma",
        "th_name","recalc_n",
        "precision","recall","f1","accuracy",
        "seg_det","seg_total","score"
    ])
    for i, combo in enumerate(done):
        import re
        m = re.search(r"r(\d+(?:\.\d+)?)$", combo.get("th_name",""))
        recalc_n = m.group(1) if m else "—"
        writer.writerow([
            i+1, combo.get("arch",""), combo.get("hidden",""),
            combo.get("seq_len",""), combo.get("epochs",""),
            combo.get("lr",""), combo.get("ewma",""),
            combo.get("th_name",""), recalc_n,
            round((combo.get("precision") or 0)*100, 1),
            round((combo.get("recall")    or 0)*100, 1),
            round((combo.get("f1")        or 0)*100, 1),
            round((combo.get("accuracy")  or 0)*100, 1),
            combo.get("n_detected_segs",0),
            combo.get("n_segments",0),
            round((combo.get("score")     or 0)*100, 1),
        ])

    buf.seek(0)
    from django.http import HttpResponse
    resp = HttpResponse(buf.read(), content_type="text/csv")
    resp["Content-Disposition"] = f'attachment; filename="grid_search_{job_id}.csv"'
    return resp
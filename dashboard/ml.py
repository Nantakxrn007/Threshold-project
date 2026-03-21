# dashboard/ml.py
"""
ML code ทั้งหมด: models, pipeline, threshold evaluator
ไม่มี dependency กับ Django — import ได้จากทุกที่
"""
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

warnings.filterwarnings("ignore")

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
FEATS = ["conductivity", "pH", "temperature", "voltage"]
NORMAL_RUNS = ["NOV_6_3", "NOV_12_6", "NOV_28_1"]
ANOMALY_RUNS = ["NOV_6_5", "NOV_14_4", "NOV_20_1", "NOV_26_8"]
ALL_RUNS = NORMAL_RUNS + ANOMALY_RUNS
LABEL_COLS = ["Anomaly V_filled", "Anomaly C_filled", "Anomaly P_filled", "Anomaly T_filled"]

# ── Data Preparation ─────────────────────────────────────────────────────────

def load_and_clean_data(file_path='data_cleaned3.csv', max_gap=30) -> pd.DataFrame:
    """
    โหลดข้อมูลจริงและจัดการ Label (Binary & Gap Filling) สำหรับทุก Sensor และทุก Run
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    sensors = ['V', 'C', 'P', 'T']
    
    # 1. สร้าง Binary Label (_bin): N, Unknown -> 0 | อื่นๆ -> 1
    for s in sensors:
        target_col = f"Anomaly {s}"
        bin_col = f"{target_col}_bin"
        filled_col = f"{target_col}_filled"
        
        # แปลงเป็นตัวเลข 0, 1
        df[bin_col] = np.where(df[target_col].isin(['N', 'Unknown']), 0, 1)
        
        # 2. ทำ Gap Filling (_filled) แยกตามแต่ละ run_id
        df[filled_col] = df[bin_col].copy()
        
        for run_id, group in df.groupby('run_id'):
            series_bin = group[bin_col].copy()
            is_zero = (series_bin == 0)
            # จัดกลุ่มเลข 0 ที่ติดกัน
            group_ids = (is_zero != is_zero.shift()).cumsum()
            
            for _, sub_group in series_bin.groupby(group_ids):
                # ถ้าเป็นกลุ่ม 0 ที่ยาว <= max_gap และมี 1 ขนาบข้าง
                if sub_group.iloc[0] == 0 and len(sub_group) <= max_gap:
                    idx = sub_group.index
                    if idx[0] > series_bin.index[0] and idx[-1] < series_bin.index[-1]:
                        if series_bin.loc[idx[0]-1] == 1 and series_bin.loc[idx[-1]+1] == 1:
                            df.loc[idx, filled_col] = 1
                            
    return df

def get_training_data() -> pd.DataFrame:
    """
    เรียกใช้ฟังก์ชัน clean ข้อมูล และส่งออก DataFrame ที่พร้อมใช้งาน
    """
    # โหลดและคลีน Label ทั้งหมด
    df = load_and_clean_data('data_cleaned7.csv', max_gap=30)
    df = df[df['run_id'].isin(ALL_RUNS)]
    # กรองเอาเฉพาะ run_id ที่เรากำหนดไว้ใน Constants (ถ้าต้องการ)
    # df = df[df['run_id'].isin(ALL_RUNS)]
    
    return df


# ── Models ───────────────────────────────────────────────────────────────────

class PlainAE(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=16, **_):
        super().__init__()
        self.seq_len = seq_len
        self.n_feat  = n_feat
        d = n_feat * seq_len
        self.enc = nn.Sequential(
            nn.Linear(d, hidden * 4), nn.ReLU(),
            nn.Linear(hidden * 4, hidden), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.ReLU(),
            nn.Linear(hidden * 4, d),
        )

    def forward(self, x):
        b = x.shape[0]
        z = self.enc(x.view(b, -1))
        return self.dec(z).view(b, self.seq_len, self.n_feat), z


class PlainLSTM(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=16, layers=1, **_):
        super().__init__()
        self.seq_len = seq_len
        self.lstm    = nn.LSTM(n_feat, hidden, layers, batch_first=True)
        self.out     = nn.Linear(hidden, n_feat)

    def forward(self, x):
        o, (hn, _) = self.lstm(x)
        return self.out(o), hn[-1]


class LSTMAE(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=16, layers=1, **_):
        super().__init__()
        self.seq_len = seq_len
        self.enc = nn.LSTM(n_feat,   hidden, layers, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (hn, _) = self.enc(x)
        z = hn[-1]
        o, _ = self.dec(z.unsqueeze(1).repeat(1, self.seq_len, 1))
        return self.out(o), z


class GRUAE(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=16, layers=1, **_):
        super().__init__()
        self.seq_len = seq_len
        self.enc = nn.GRU(n_feat,   hidden, layers, batch_first=True)
        self.dec = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, hn = self.enc(x)
        z = hn[-1]
        o, _ = self.dec(z.unsqueeze(1).repeat(1, self.seq_len, 1))
        return self.out(o), z


class CNNLSTMAE(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=16, layers=1, cnn_filters=32, **_):
        super().__init__()
        self.seq_len = seq_len
        self.conv = nn.Conv1d(n_feat, cnn_filters, kernel_size=3, padding=1)
        self.enc  = nn.LSTM(cnn_filters, hidden, layers, batch_first=True)
        self.dec  = nn.LSTM(hidden,      hidden, layers, batch_first=True)
        self.out  = nn.Linear(hidden, n_feat)

    def forward(self, x):
        xc = torch.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        _, (hn, _) = self.enc(xc)
        z = hn[-1]
        o, _ = self.dec(z.unsqueeze(1).repeat(1, self.seq_len, 1))
        return self.out(o), z


class _Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.w = nn.Linear(hidden, 1)

    def forward(self, enc_out):
        # enc_out: (B, T, H)
        weights = torch.softmax(self.w(enc_out).squeeze(-1), dim=1)  # (B, T)
        context = (enc_out * weights.unsqueeze(-1)).sum(dim=1)        # (B, H)
        return context, weights


class LSTMAttentionAE(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=16, layers=1, **_):
        super().__init__()
        self.seq_len = seq_len
        self.enc  = nn.LSTM(n_feat, hidden, layers, batch_first=True)
        self.attn = _Attention(hidden)
        self.dec  = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out  = nn.Linear(hidden, n_feat)

    def forward(self, x):
        enc_out, _ = self.enc(x)
        z, _       = self.attn(enc_out)
        o, _       = self.dec(z.unsqueeze(1).repeat(1, self.seq_len, 1))
        return self.out(o), z


MODEL_MAP = {
    "LSTM-AE":           LSTMAE,
    "GRU-AE":            GRUAE,
    "CNN-LSTM-AE":       CNNLSTMAE,
    "Plain-AE":          PlainAE,
    "Plain-LSTM":        PlainLSTM,
    "LSTM-Attention-AE": LSTMAttentionAE,
}


# ── Pipeline ─────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(
        self,
        model_type: str,
        seq_len:    int,
        ewma:       float,
        epochs:     int,
        batch_size: int,
        hidden:     int,
        layers:     int,
        lr:         float,
    ):
        self.seq_len    = seq_len
        self.ewma       = ewma
        self.epochs     = epochs
        self.batch_size = batch_size
        self.scaler     = StandardScaler()
        # Auto-detect: CUDA → MPS (Apple) → CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.loss_history: list[float] = []

        cls         = MODEL_MAP[model_type]
        self.model  = cls(len(FEATS), seq_len, hidden=hidden, layers=layers).to(self.device)
        self.opt    = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # ── helpers ──────────────────────────────────────────────────

    def _make_sequences(self, arr: np.ndarray) -> np.ndarray:
        return np.array([
            arr[i : i + self.seq_len]
            for i in range(len(arr) - self.seq_len + 1)
        ])

    # ── public ───────────────────────────────────────────────────

    def train(self, df_normal: pd.DataFrame, on_epoch=None) -> None:
        self.scaler.fit(df_normal[FEATS])

        all_seqs = []
        for rid in df_normal["run_id"].unique():
            scaled = self.scaler.transform(
                df_normal[df_normal["run_id"] == rid][FEATS]
            )
            seqs = self._make_sequences(scaled)
            if len(seqs):
                all_seqs.append(seqs)

        X  = torch.FloatTensor(np.vstack(all_seqs)).to(self.device)
        dl = DataLoader(
            TensorDataset(X, X),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model.train()
        self.loss_history = []

        for _ in range(self.epochs):
            total = 0.0
            for bx, _ in dl:
                self.opt.zero_grad()
                recon, _ = self.model(bx)
                loss = self.loss_fn(recon, bx)
                loss.backward()
                self.opt.step()
                total += loss.item()
            self.loss_history.append(total / len(dl))
            if on_epoch:
                on_epoch(len(self.loss_history), self.loss_history[-1])

    def evaluate(self, df: pd.DataFrame, agg: str = "mean"):
        """
        agg: วิธีรวม per-sensor error เป็น overall_error
          "mean" — ค่าเฉลี่ย (default เดิม, anomaly ถูก dilute)
          "max"  — ค่าสูงสุด (sensor ใดผิดปกติก็โผล่ขึ้นมาเลย)
          "l2"   — Euclidean norm (ขยาย outlier ด้วย squaring)
        """
        self.model.eval()
        err_rows, z_vecs, z_labels = [], [], []

        for rid in df["run_id"].unique():
            dfr    = df[df["run_id"] == rid].reset_index(drop=True)
            scaled = self.scaler.transform(dfr[FEATS])
            seqs   = self._make_sequences(scaled)
            if not len(seqs):
                continue

            with torch.no_grad():
                recon, z = self.model(torch.FloatTensor(seqs).to(self.device))

            z_vecs.append(z.cpu().numpy())
            z_labels.extend([rid] * len(z))

            mae = np.abs(seqs[:, -1, :] - recon.cpu().numpy()[:, -1, :])
            df_mae = (
                pd.DataFrame(mae, columns=FEATS)
                .ewm(alpha=self.ewma, adjust=False)
                .mean()
            )

            # ── Aggregation method ──────────────────────────────────
            if agg == "max":
                df_mae["overall_error"] = df_mae[FEATS].max(axis=1)
            elif agg == "l2":
                df_mae["overall_error"] = np.sqrt((df_mae[FEATS] ** 2).sum(axis=1))
            else:  # "mean"
                df_mae["overall_error"] = df_mae[FEATS].mean(axis=1)

            # Pad the first (seq_len - 1) rows with NaN
            pad   = pd.DataFrame(np.nan, index=range(self.seq_len - 1), columns=df_mae.columns)
            dfull = pd.concat([pad, df_mae], ignore_index=True)
            dfull["run_id"] = rid
            err_rows.append(dfull)

        return pd.concat(err_rows, ignore_index=True), np.vstack(z_vecs), z_labels


# ── Threshold Evaluator ──────────────────────────────────────────────────────

class ThresholdEvaluator:
    def __init__(self, normal_errors: np.ndarray):
        self.normal_errors = normal_errors

    def calculate(self, err_series: pd.Series, cfg: dict):
        err = err_series.fillna(0).values
        n   = len(err)

        # ── TH1: Sliding Window Percentile ───────────────────────
        # th1_mode: "static" = คำนวณจาก normal_errors ทั้งหมดครั้งเดียว (เส้นตรง)
        #           "sliding" = คำนวณจาก window ก่อนหน้า N points เท่านั้น (realistic)
        pct_val   = cfg.get("th1_pct",    99.0)
        th1_mode  = cfg.get("th1_mode",   "sliding")
        win1      = max(int(cfg.get("th1_win",    100)), 1)
        recalc1   = max(int(cfg.get("th1_recalc",  10)), 1)

        th1 = np.zeros(n)
        if th1_mode == "static":
            # เดิม: ใช้ normal_errors เป็น baseline (ไม่ใช้ future data)
            base_val = float(np.nanpercentile(self.normal_errors, pct_val))
            th1[:] = base_val
        else:
            # Sliding: percentile จาก window ก่อนหน้าเท่านั้น
            # warmup: ใช้ normal_errors แทนจนกว่าจะมี data พอ
            cur1 = float(np.nanpercentile(self.normal_errors, pct_val))
            for i in range(n):
                if i > 0 and i % recalc1 == 0:
                    w    = err[max(0, i - win1) : i]
                    cur1 = float(np.nanpercentile(w, pct_val))
                th1[i] = cur1

        # ── TH2: Sliding window Mu + α·Std ───────────────────────
        th2    = np.zeros(n)
        alpha2 = cfg.get("th2_alpha",  3.5)
        win2   = max(int(cfg.get("th2_win",   80)), 1)
        recalc = max(int(cfg.get("th2_recalc", 50)), 1)
        mu, sd = np.mean(err[:win2]), np.std(err[:win2])
        cur2   = mu + alpha2 * sd
        for i in range(n):
            if i >= win2 and i % recalc == 0:
                mu, sd = np.mean(err[i - win2 : i]), np.std(err[i - win2 : i])
                cur2   = mu + alpha2 * sd
            th2[i] = cur2

        # ── TH3: Adaptive-z ───────────────────────────────────────
        th3     = np.zeros(n)
        zlo     = cfg.get("th3_zmin",   2.0)
        zhi     = cfg.get("th3_zmax",  10.0)
        win3    = max(int(cfg.get("th3_win",   80)), 1)
        recalc3 = max(int(cfg.get("th3_recalc", 1)), 1)
        cur3    = np.mean(err[:win3]) + zlo * np.std(err[:win3])
        for i in range(n):
            if i >= win3 and (i - win3) % recalc3 == 0:
                w          = err[i - win3 : i]
                mu_a, sd_a = np.mean(w), np.std(w)
                below  = w[w < cur3] if (w < cur3).any() else w
                dm     = mu_a - np.mean(below)
                ds     = sd_a - np.std(below)
                ei     = np.where(w > cur3)[0]
                e_seqs = np.sum(np.diff(ei) > 1) + 1 if len(ei) > 0 else 0
                denom  = max(1, len(ei) + e_seqs)
                z_val  = np.clip(
                    (dm / max(mu_a, 1e-6) + ds / max(sd_a, 1e-6)) / denom,
                    zlo, zhi,
                )
                cur3 = mu_a + z_val * sd_a
            th3[i] = cur3

        # ── TH4: Entropy-lock ─────────────────────────────────────
        th4       = np.zeros(n)
        alpha4    = cfg.get("th4_alpha", 3.5)
        win4      = max(int(cfg.get("th4_win",    150)), 1)
        cons_req  = max(int(cfg.get("th4_cons",     5)), 1)
        eth       = cfg.get("th4_eth",         0.95)
        recalc4   = max(int(cfg.get("th4_recalc",   1)), 1)
        locked_n, cons_viol, Dh1, h_prev = None, 0, None, 0.0
        cur_th4   = err[0] + 0.1 if n > 0 else 0.1

        for k in range(1, n):
            if k % recalc4 == 0:
                if locked_n is None and k > 10:
                    wd = err[max(0, k - win4) : k]
                    if len(wd) > 2:
                        counts, _ = np.histogram(wd, bins="auto")
                        probs     = counts[counts > 0] / len(wd)
                        hk        = entropy(probs, base=2)
                        Dhk       = (hk - h_prev) / 2.0
                        h_prev    = hk
                        if k == 20:
                            Dh1 = abs(Dhk)
                        if Dh1 and k > 20:
                            if abs(Dhk) < (1 - eth) * Dh1:
                                cons_viol += 1
                            else:
                                cons_viol = 0
                            if cons_viol >= cons_req:
                                locked_n = min(k, win4)

                actual_win = locked_n if locked_n else min(k, win4)
                w = err[max(0, k - actual_win) : k]
                cur_th4 = (
                    np.mean(w) + alpha4 * max(np.std(w), 0.01)
                    if len(w) > 2
                    else err[k] + 0.1
                )
            th4[k] = cur_th4

        return th1, th2, th3, th4

# ── Metrics ───────────────────────────────────────────────────────────────────

TH_NAMES = ["P99 Static", "Sliding Mu+αStd", "Adaptive-z", "Entropy-lock"]


def _get_anomaly_segments(y_true: np.ndarray) -> list:
    """
    หาช่วง anomaly segments จาก y_true (0/1 array)
    Returns: list of (start, end) inclusive index pairs
    """
    segments = []
    in_seg = False
    for i, v in enumerate(y_true):
        if v == 1 and not in_seg:
            seg_start = i
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(y_true) - 1))
    return segments


def _point_adjust(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Point-Adjusted prediction:
    ถ้า model ตรวจเจอ anomaly อย่างน้อย 1 point ในช่วง anomaly segment
    → นับว่าทุก point ในช่วงนั้นเป็น TP หมด (y_pred_adj = 1 ทั้งช่วง)

    หมายเหตุ: _filled label ที่ใช้ทำ Gap Filling ≤ 30 points ไปแล้ว
    ทำให้ segment ที่ห่างกัน ≤ 30 points ถูก merge เป็น segment เดียวก่อน
    ดังนั้น Point-Adjusted นี้จึงทำงานบน cleaned segment ที่ถูกต้องแล้ว
    """
    y_adj = y_pred.copy()
    for seg_start, seg_end in _get_anomaly_segments(y_true):
        # ถ้าตรวจเจออย่างน้อย 1 point ใน segment → adjust ทั้งช่วง
        if y_pred[seg_start:seg_end + 1].any():
            y_adj[seg_start:seg_end + 1] = 1
    return y_adj


def compute_metrics_from_error(df_err: pd.DataFrame, df_original: pd.DataFrame, cfg: dict) -> tuple:
    """
    คำนวณ Confusion Matrix metrics (Point-Adjusted) ทุก run × ทุก threshold method
    Returns: (results_dict, has_labels)
      results_dict = { run_id: { th_name: { tp,fp,tn,fn,precision,recall,f1,accuracy,flagged,pct,
                                             pa_tp,pa_fp,pa_tn,pa_fn,n_segments,n_detected_segs } } }
    """
    has_labels = all(c in df_original.columns for c in LABEL_COLS)

    normal_errors = df_err[df_err["run_id"].isin(NORMAL_RUNS)]["overall_error"].dropna().values
    evaluator     = ThresholdEvaluator(normal_errors)
    results       = {}

    for run_id in df_err["run_id"].unique():
        dfr        = df_err[df_err["run_id"] == run_id].reset_index(drop=True)
        th1, th2, th3, th4 = evaluator.calculate(dfr["overall_error"], cfg)
        err_vals   = dfr["overall_error"].fillna(0).values
        valid_mask = ~dfr["overall_error"].isna().values

        if has_labels:
            df_run     = df_original[df_original["run_id"] == run_id].reset_index(drop=True)
            y_true_raw = (df_run[LABEL_COLS].max(axis=1).values > 0).astype(int)
            # align length to error array
            n = len(err_vals)
            if len(y_true_raw) >= n:
                y_true = y_true_raw[:n]
            else:
                y_true = np.concatenate([y_true_raw, np.zeros(n - len(y_true_raw), dtype=int)])
        else:
            y_true = None

        run_result = {}
        for th_name, th_vals in zip(TH_NAMES, [th1, th2, th3, th4]):
            y_pred  = (err_vals > th_vals).astype(int)
            flagged = int(y_pred.sum())
            pct     = round(100 * flagged / max(len(y_pred), 1), 1)
            m       = dict(flagged=flagged, pct=pct)

            if y_true is not None:
                # ── Point-Adjusted ────────────────────────────────────
                y_adj = _point_adjust(y_pred, y_true)

                # apply valid_mask (ignore NaN padded rows)
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

                # ── segment-level stats ───────────────────────────────
                # นับ segments จาก y_true (overall = max ของทุก sensor หลัง gap fill)
                # ใช้ y_true ตรงๆ ไม่ต้อง union per-sensor เพราะ y_true ทำไปแล้ว
                segments = _get_anomaly_segments(y_true)
                n_segs   = len(segments)
                detected = sum(1 for s, e in segments if y_pred[s:e+1].any())

                m.update(dict(
                    tp=tp, fp=fp, tn=tn, fn=fn,
                    precision=round(prec, 4),
                    recall=round(rec, 4),
                    f1=round(f1, 4),
                    accuracy=round(acc, 4),
                    n_segments=n_segs,
                    n_detected_segs=detected,
                ))

            run_result[th_name] = m
        results[run_id] = run_result

    return results, has_labels

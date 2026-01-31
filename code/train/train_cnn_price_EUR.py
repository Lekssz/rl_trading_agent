# EURUSD 30m UTC — 1D CNN price feature extractor
# Outputs: prob_up_1h + embeddings z1...zK
# Train: 2019–2021 | Signal Test: 2022 | Forward: 2023

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# =====================================================
# PROJECT ROOT
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EUR_CSV = PROJECT_ROOT / "data/processed/ohlcv/oanda_EURUSD_M30_2019-2023_processed.csv"
OUTDIR  = PROJECT_ROOT / "data/model/cnn_price_eurusd"

# =====================================================
# Utils
# =====================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

def bce_pos_weight(y):
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return max(neg / max(1.0, pos), 1.0)

# =====================================================
# Args
# =====================================================
def parse_args():
    ap = argparse.ArgumentParser("EURUSD CNN price encoder")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--lookback", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--bars-per-day", type=int, default=48)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--dilations", type=str, default="1,2,4,8,16")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# =====================================================
# Feature engineering
# =====================================================
def build_features(df, bars_per_day):
    df = df.sort_values("time").drop_duplicates("time").copy()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["log_close"] = np.log(df["close"].replace(0, np.nan))
    df["ret"] = df["log_close"].diff().fillna(0.0)

    df["rng"] = (df["high"] - df["low"]).fillna(0.0)
    df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    df["log_vol"] = np.log1p(df["volume"].clip(lower=0))

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    w = int(bars_per_day)
    df["rv_1d"] = (df["ret"].pow(2).rolling(w, min_periods=1).sum()).pow(0.5).fillna(0.0)

    ts = pd.to_datetime(df["time"], utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    df["hour_cos"] = np.cos(2*np.pi*hour/24.0)

    features = [
        "ret","rng","ohlc4","log_vol",
        "atr14","rv_1d",
        "hour_sin","hour_cos"
    ]

    X = df[features].to_numpy(np.float32)
    times = ts.to_numpy()
    return df, X, times

# =====================================================
# Labels & windows
# =====================================================
def make_windows(X, ret, times, lookback, horizon):
    xs, ys, tlist = [], [], []
    for t in range(lookback - 1, len(X) - horizon):
        w = X[t-lookback+1:t+1]
        if np.isnan(w).any():
            continue
        xs.append(w.T)
        ys.append((ret[t+1:t+1+horizon].sum() > 0))
        tlist.append(times[t])
    return np.stack(xs), np.array(ys, np.float32), np.array(tlist)

# =====================================================
# Dataset
# =====================================================
class SeqDS(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = ((X - mean[:, None]) / (std[:, None] + 1e-8)).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.as_tensor(self.X[i], dtype=torch.float32),
            torch.as_tensor(self.y[i], dtype=torch.float32)
        )

# =====================================================
# Model
# =====================================================
class ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.act(self.c2(y))
        return y + x

class Encoder(nn.Module):
    def __init__(self, in_ch, emb_dim, dilations):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(64, d) for d in dilations])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, emb_dim)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)

class Head(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)

# =====================================================
# Training
# =====================================================
def train_val(enc, head, dl_tr, dl_va, epochs, pos_weight, patience, device):
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(head.parameters()),
        lr=3e-4, weight_decay=1e-2
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_auc, wait = -1.0, 0

    for ep in range(1, epochs + 1):
        enc.train(); head.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(head(enc(xb)), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        enc.eval(); head.eval()
        yt, ys = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                p = torch.sigmoid(head(enc(xb.to(device)))).cpu().numpy()
                ys.append(p.reshape(-1).astype(np.float32))
                yt.append(yb.numpy().astype(np.float32))
        ys = np.concatenate(ys).astype(np.float64)   # prediction scores
        yt = np.concatenate(yt).astype(np.int32)    # true labels (0/1)

        auc = roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else 0.5
        print(f"Epoch {ep:02d} | Val AUC={auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading EURUSD CSV:", EUR_CSV)
    df = pd.read_csv(EUR_CSV)
    df = df.rename(columns={args.time_col: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)

    df_feat, X_raw, times = build_features(df, args.bars_per_day)
    ret = df_feat["ret"].to_numpy(np.float32)

    Xw, yb, times_w = make_windows(X_raw, ret, times, args.lookback, args.horizon)
    print(f"[EURUSD] windows={Xw.shape}, positives={yb.sum():.0f}/{len(yb)}")

    test_ts = pd.Timestamp(args.test_start, tz="UTC")
    idx_test = times_w >= test_ts
    idx_tv = ~idx_test

    valN = int(0.15 * idx_tv.sum())
    train_idx = np.where(idx_tv)[0][:-valN]
    val_idx   = np.where(idx_tv)[0][-valN:]

    mean = Xw[train_idx].mean(axis=(0,2))
    std  = Xw[train_idx].std(axis=(0,2))

    dl_tr = DataLoader(SeqDS(Xw[train_idx], yb[train_idx], mean, std),
                       batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(SeqDS(Xw[val_idx], yb[val_idx], mean, std),
                       batch_size=args.batch_size, num_workers=0)

    dilations = tuple(int(x) for x in args.dilations.split(","))
    enc = Encoder(Xw.shape[1], args.emb_dim, dilations).to(device)
    head = Head(args.emb_dim).to(device)

    posw = bce_pos_weight(yb[train_idx])
    train_val(enc, head, dl_tr, dl_va,
              args.epochs, posw, args.patience, device)

    enc.eval(); head.eval()
    all_emb, all_prob = [], []
    with torch.no_grad():
        for xb, _ in DataLoader(SeqDS(Xw, yb, mean, std),
                                batch_size=args.batch_size, num_workers=0):
            z = enc(xb.to(device))
            p = torch.sigmoid(head(z))
            all_emb.append(z.cpu().numpy())
            all_prob.append(p.cpu().numpy())

    out = pd.DataFrame({
        "time": pd.to_datetime(times_w),
        "instrument": "EURUSD",
        "prob_up_1h": np.concatenate(all_prob)
    })
    emb = np.vstack(all_emb)
    for i in range(emb.shape[1]):
        out[f"z{i+1}"] = emb[:, i]

    out_path = OUTDIR / "cnn_price_embeddings_EURUSD.csv"
    out.to_csv(out_path, index=False)
    print("Saved embeddings →", out_path)

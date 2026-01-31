# train_cnn_price_btc.py
# BTCUSDT 30m UTC — 1D CNN price feature extractor
# Outputs: prob_up_1h + embeddings z1...zK for every bar
# Train=2019–2021, Test=2022, Forward=2023

import argparse, os, json, random
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# -----------------------------
# Section 0 — Utils
# -----------------------------
def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def spearman_ic(y_true, y_score):
    r1 = pd.Series(y_true).rank().to_numpy()
    r2 = pd.Series(y_score).rank().to_numpy()
    if r1.std() == 0 or r2.std() == 0:
        return 0.0
    return float(np.corrcoef(r1, r2)[0,1])

def bce_pos_weight(y):
    pos = float(y.sum()); neg = float(len(y) - pos)
    return max(neg / max(1.0, pos), 1.0)

# -----------------------------
# Section 1 — CLI arguments
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("BTC 1D CNN price model")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--instrument", default="BTCUSDT")
    ap.add_argument("--lookback", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--bars-per-day", type=int, default=48)

    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--dilations", type=str, default="1,2,4,8,16")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)

    ap.add_argument("--outdir", default="data/model/cnn_price")
    ap.add_argument("--test-start", default="2022-01-01")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))
    return ap.parse_args()

# -----------------------------
# Section 2 — Feature engineering
# -----------------------------
def build_features(df, bars_per_day):
    df = df.sort_values("time").drop_duplicates("time").copy()

    # base columns
    for c in ["open","high","low","close"]:
        if c not in df.columns:
            raise ValueError(f"Missing OHLC column: {c}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # engineered features
    df["log_close"] = np.log(df["close"].replace(0, np.nan))
    df["ret"] = df["log_close"].diff().fillna(0.0)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["clv"] = (((df["close"]-df["low"]) - (df["high"]-df["close"])) / rng).fillna(0).replace([np.inf,-np.inf],0)
    df["ohlc4"] = (df["open"]+df["high"]+df["low"]+df["close"])/4.0
    df["rng"] = (df["high"] - df["low"]).fillna(0.0)
    df["log_vol"] = np.log1p(df["volume"].clip(lower=0))

    # ATR14
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    # realized vol (1 day = 48 bars)
    w = int(bars_per_day)
    df["rv_1d"] = (df["ret"].pow(2).rolling(w, min_periods=1).sum()).pow(0.5).fillna(0.0)

    # zscore of close (20 bars)
    m = df["close"].rolling(20, min_periods=1).mean()
    s = df["close"].rolling(20, min_periods=1).std().replace(0, np.nan)
    df["zclose20"] = ((df["close"] - m)/s).replace([np.inf,-np.inf],0).fillna(0)

    # time features
    ts = pd.to_datetime(df["time"], utc=True)
    hour = ts.dt.hour + ts.dt.minute/60.0
    df["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    df["hour_cos"] = np.cos(2*np.pi*hour/24.0)
    dow = ts.dt.weekday
    df["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    df["dow_cos"] = np.cos(2*np.pi*dow/7.0)

    features = [
        "ret","rng","clv","ohlc4","log_vol",
        "atr14","rv_1d","zclose20",
        "hour_sin","hour_cos","dow_sin","dow_cos"
    ]

    X = df[features].to_numpy(np.float32)
    times = pd.to_datetime(df["time"], utc=True).to_numpy()
    return df, X, features, times

# -----------------------------
# Section 3 — Labels & windows
# -----------------------------
def build_labels_from_returns(ret, horizon):
    N = len(ret)
    fut = np.full(N, np.nan, np.float32)
    for i in range(N - horizon):
        fut[i] = ret[i+1:i+1+horizon].sum()
    y = (fut > 0).astype(np.float32)
    return fut, y

def make_windows(X, ret, times, lookback, horizon):
    fut, y = build_labels_from_returns(ret, horizon)
    xs, ys, idxs, tlist = [], [], [], []

    for t in range(lookback-1, len(X)-horizon):
        w = X[t-lookback+1:t+1]
        if np.isnan(w).any():
            continue
        xs.append(w.T)       # [C,L]
        ys.append(y[t])
        idxs.append(t)
        tlist.append(times[t])

    return np.stack(xs).astype(np.float32), np.array(ys), np.array(idxs), np.array(tlist)

# -----------------------------
# Section 4 — Dataset loader
# -----------------------------
class SeqDS(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = (X - mean[:,None])/(std[:,None] + 1e-8)
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

# -----------------------------
# Section 5 — Model definition
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.act = nn.ReLU()
        self.ln  = nn.LayerNorm(ch)

    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.act(self.c2(y))
        y = y + x
        y = y.transpose(1,2)
        y = self.ln(y)
        return y.transpose(1,2)

class Encoder(nn.Module):
    def __init__(self, in_ch, emb_dim=64, dilations=(1,2,4,8,16)):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(64, d) for d in dilations])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, emb_dim)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        z = self.proj(h)
        return z

class Head(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.m = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, z):
        return self.m(z).squeeze(-1)

# -----------------------------
# Section 6 — Training loop
# -----------------------------
def train_val(enc, head, dl_tr, dl_val, epochs, pos_weight, patience, device, lr=3e-4, wd=1e-2):
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_auc, best, wait = -1.0, None, 0

    for ep in range(1, epochs+1):
        enc.train(); head.train()
        tot = 0.0

        for xb,yb in dl_tr:
            xb,yb = xb.to(device), yb.to(device)
            logit = head(enc(xb))
            loss = crit(logit, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)

        # validation
        enc.eval(); head.eval()
        yt, ys = [], []
        with torch.no_grad():
            for xb,yb in dl_val:
                p = torch.sigmoid(head(enc(xb.to(device)))).cpu().numpy()
                yt.append(yb.numpy()); ys.append(p)
        yt = np.concatenate(yt); ys = np.concatenate(ys)

        auc = roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else 0.5
        ap = average_precision_score(yt, ys)
        ic = spearman_ic(yt, ys)
        print(f"Epoch {ep:02d} | loss={tot/len(dl_tr.dataset):.4f} | AUC={auc:.3f} AP={ap:.3f} IC={ic:.3f}")

        if auc > best_auc:
            best_auc = auc
            best = (enc.state_dict(), head.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best is not None:
        enc.load_state_dict(best[0])
        head.load_state_dict(best[1])

    return best_auc

# -----------------------------
# Section 7 — Test evaluation
# -----------------------------
def evaluate(enc, head, dl_te, device):
    enc.eval(); head.eval()
    yt, ys = [], []
    with torch.no_grad():
        for xb,yb in dl_te:
            p = torch.sigmoid(head(enc(xb.to(device)))).cpu().numpy()
            yt.append(yb.numpy()); ys.append(p)
    yt = np.concatenate(yt); ys = np.concatenate(ys)

    auc = roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else 0.5
    ap  = average_precision_score(yt, ys)
    ic  = spearman_ic(yt, ys)
    return {"AUC": float(auc), "AP": float(ap), "IC": float(ic)}

# -----------------------------
# Section 8 — Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)

    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    if args.time_col not in df.columns:
        raise SystemExit(f"Missing time column '{args.time_col}'")
    df = df.rename(columns={args.time_col: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.dropna(subset=["time"])

    # Features
    df_feat, X_raw, feat_names, times = build_features(df, args.bars_per_day)
    ret = df_feat["ret"].to_numpy(np.float32)

    # Windows
    Xw, yb, idxs, times_w = make_windows(X_raw, ret, times, args.lookback, args.horizon)
    C, L = Xw.shape[1], Xw.shape[2]

    print(f"[BTCUSDT] windows={Xw.shape}, positives={yb.sum():.0f}/{len(yb)} ({100*yb.mean():.1f}%)")

    # Time split
    test_ts = pd.Timestamp(args.test_start, tz="UTC")
    tw_idx = pd.DatetimeIndex(times_w)

    mask_test = tw_idx >= test_ts
    test_idx = np.where(mask_test)[0]
    tv_idx = np.where(~mask_test)[0]

    valN = int(0.15 * len(tv_idx))
    train_idx = tv_idx[:-valN]
    val_idx   = tv_idx[-valN:]

    # Normalize (train only)
    Xtr = Xw[train_idx]
    mean = Xtr.mean(axis=(0,2)).astype(np.float32)
    std  = Xtr.std(axis=(0,2)).astype(np.float32)

    # DataLoaders
    ds_tr = SeqDS(Xw[train_idx], yb[train_idx], mean, std)
    ds_va = SeqDS(Xw[val_idx],   yb[val_idx],   mean, std)
    ds_te = SeqDS(Xw[test_idx],  yb[test_idx],  mean, std)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    dilations = tuple(int(x) for x in args.dilations.split(","))
    enc  = Encoder(C, emb_dim=args.emb_dim, dilations=dilations).to(device)
    head = Head(args.emb_dim).to(device)

    # Train
    posw = bce_pos_weight(yb[train_idx])
    best_auc = train_val(enc, head, dl_tr, dl_va, args.epochs, posw, args.patience, device)

    # Test
    test_metrics = evaluate(enc, head, dl_te, device)
    print(f"TEST | AUC={test_metrics['AUC']:.3f} | AP={test_metrics['AP']:.3f} | IC={test_metrics['IC']:.3f}")

    # Export embeddings + probabilities
    all_ds = SeqDS(Xw, yb, mean, std)
    all_dl = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_emb, all_prob = [], []
    with torch.no_grad():
        for xb,_ in all_dl:
            z = enc(xb.to(device))
            p = torch.sigmoid(head(z))
            all_emb.append(z.cpu().numpy())
            all_prob.append(p.cpu().numpy())

    all_emb = np.vstack(all_emb)
    all_prob = np.concatenate(all_prob)

    # Splits
    split = np.array(["train"]*len(yb))
    split[val_idx] = "val"
    split[test_idx] = "test"

    # Save embeddings
    out = pd.DataFrame({
        "time": pd.to_datetime(times_w),
        "instrument": args.instrument,
        "split": split,
        "prob_up_1h": all_prob
    })
    for i in range(all_emb.shape[1]):
        out[f"z{i+1}"] = all_emb[:, i]

    out_path = outdir / f"cnn_price_embeddings_{args.instrument}.csv"
    out.to_csv(out_path, index=False)

    # Save metrics
    with open(outdir / f"metrics_{args.instrument}.json","w") as f:
        json.dump({
            "val_best_auc": float(best_auc),
            "test_auc": test_metrics["AUC"],
            "test_ap": test_metrics["AP"],
            "test_ic": test_metrics["IC"],
            "lookback": args.lookback,
            "horizon": args.horizon,
            "emb_dim": args.emb_dim,
            "dilations": list(dilations),
            "features": feat_names
        }, f, indent=2)

    print(f"Saved embeddings → {out_path}")

"""Sequence model — Transformer over the last K balls.

The LightGBM baseline scores each delivery independently. But cricket has
real momentum: a bowler who's been hit for 16 in two overs is more likely
to leak runs on the next ball, an opening pair off to a flier shifts the
match score distribution, etc.

This module learns a small Transformer encoder over a rolling window of
the last `SEQ_LEN` deliveries (per innings) and outputs the same two heads
as the LightGBM model: a 7-class softmax over runs and a single sigmoid
over wicket. Models are saved to data/models/sequence.pt.

Inference: pass a list of state dicts representing the last few balls. The
model handles padding for the start of an innings (mask + zeros).

Notes:
  * Trains on whatever fits in memory; for very large datasets, tile by
    match into shards.
  * Categoricals (batter, bowler, venue) are learned embeddings — unseen
    values map to id 0 ("OOV").
  * Calibration: same isotonic mapping from `model/calibrate.py` is fit on
    a held-out 10% slice and applied automatically by `predict_sequence`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from . import calibrate as C
from . import features as F
from .train import RUN_BUCKETS

# --- hyperparameters -------------------------------------------------------

SEQ_LEN     = 12       # window of past balls fed to the encoder
EMB_DIM     = 16       # embedding dimension for batter/bowler/venue
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
DROPOUT     = 0.1
BATCH       = 512
EPOCHS      = 8
LR          = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP   = 1.0

MODEL_DIR     = Path(__file__).resolve().parent.parent / "data" / "models"
SEQ_PATH      = MODEL_DIR / "sequence.pt"
SEQ_META_PATH = MODEL_DIR / "sequence_meta.json"
SEQ_CALIB_DIR = MODEL_DIR
SEQ_RUNS_CALIB   = SEQ_CALIB_DIR / "sequence_runs_calibrators.joblib"
SEQ_WICKET_CALIB = SEQ_CALIB_DIR / "sequence_wicket_calibrator.joblib"


# --- vocab + dataset -------------------------------------------------------

def _build_vocab(values: pd.Series) -> dict:
    """0 reserved for OOV/pad; first real value is id 1."""
    unique = values.dropna().astype(str).unique().tolist()
    return {v: i + 1 for i, v in enumerate(sorted(unique))}


def _lookup(vocab: dict, x) -> int:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0
    return vocab.get(str(x), 0)


class BallSequenceDataset(Dataset):
    """One sample = the last `seq_len` balls within an innings, ending at
    the target ball. Earlier balls in the innings are zero-padded."""

    def __init__(self, df: pd.DataFrame, vocab: dict, seq_len: int = SEQ_LEN,
                 numeric_cols: list[str] | None = None) -> None:
        self.seq_len = seq_len
        self.numeric_cols = numeric_cols or F.NUMERIC
        self.bat_idx = vocab["batter"]
        self.bow_idx = vocab["bowler"]
        self.ven_idx = vocab["venue"]

        df = df.sort_values(["match_id", "innings_no", "deliveries_so_far"]).reset_index(drop=True)
        self._numeric = df[self.numeric_cols].fillna(0).to_numpy(np.float32)
        self._bat = df["batter"].map(lambda x: _lookup(self.bat_idx, x)).to_numpy(np.int64)
        self._bow = df["bowler"].map(lambda x: _lookup(self.bow_idx, x)).to_numpy(np.int64)
        self._ven = df["venue"].map(lambda x: _lookup(self.ven_idx, x)).to_numpy(np.int64)
        self._yr  = df["y_runs_bucket"].astype(int).to_numpy(np.int64)
        self._yw  = df["y_wicket"].astype(np.float32).to_numpy()

        # Per-innings start indices for fast windowing
        keys = list(zip(df["match_id"], df["innings_no"]))
        self._innings_start = np.zeros(len(df), dtype=np.int64)
        cur = 0
        prev_key = None
        for i, k in enumerate(keys):
            if k != prev_key:
                cur = i
                prev_key = k
            self._innings_start[i] = cur

    def __len__(self) -> int:
        return len(self._numeric)

    def __getitem__(self, i: int):
        start = max(self._innings_start[i], i - self.seq_len + 1)
        actual_len = i - start + 1
        pad = self.seq_len - actual_len

        num = np.zeros((self.seq_len, len(self.numeric_cols)), dtype=np.float32)
        bat = np.zeros(self.seq_len, dtype=np.int64)
        bow = np.zeros(self.seq_len, dtype=np.int64)
        ven = np.zeros(self.seq_len, dtype=np.int64)
        mask = np.zeros(self.seq_len, dtype=np.float32)

        num[pad:] = self._numeric[start:i + 1]
        bat[pad:] = self._bat[start:i + 1]
        bow[pad:] = self._bow[start:i + 1]
        ven[pad:] = self._ven[start:i + 1]
        mask[pad:] = 1.0

        return num, bat, bow, ven, mask, self._yr[i], self._yw[i]


def _collate(batch, device: torch.device):
    nums, bats, bows, vens, masks, yrs, yws = zip(*batch)
    return (
        torch.from_numpy(np.stack(nums)).to(device),
        torch.from_numpy(np.stack(bats)).to(device),
        torch.from_numpy(np.stack(bows)).to(device),
        torch.from_numpy(np.stack(vens)).to(device),
        torch.from_numpy(np.stack(masks)).to(device),
        torch.tensor(yrs, dtype=torch.long, device=device),
        torch.tensor(yws, dtype=torch.float32, device=device),
    )


# --- model -----------------------------------------------------------------

class SequenceBallModel(nn.Module):
    def __init__(self, n_numeric: int, n_batters: int, n_bowlers: int, n_venues: int,
                 d_model: int = D_MODEL, n_heads: int = N_HEADS, n_layers: int = N_LAYERS):
        super().__init__()
        self.batter_emb = nn.Embedding(n_batters + 1, EMB_DIM, padding_idx=0)
        self.bowler_emb = nn.Embedding(n_bowlers + 1, EMB_DIM, padding_idx=0)
        self.venue_emb  = nn.Embedding(n_venues + 1, EMB_DIM, padding_idx=0)

        in_dim = n_numeric + 3 * EMB_DIM
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.01)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=DROPOUT, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.runs_head   = nn.Linear(d_model, len(RUN_BUCKETS))
        self.wicket_head = nn.Linear(d_model, 1)

    def forward(self, num, bat, bow, ven, mask):
        be = self.batter_emb(bat)
        we = self.bowler_emb(bow)
        ve = self.venue_emb(ven)
        x = torch.cat([num, be, we, ve], dim=-1)
        x = self.proj(x) + self.pos_emb[:, :x.size(1)]
        kp_mask = mask == 0   # True where padded
        h = self.encoder(x, src_key_padding_mask=kp_mask)
        # use the last (most-recent, real) token: that's index -1 because we
        # pad on the LEFT, so the current ball is always at position seq_len-1
        last = h[:, -1]
        return self.runs_head(last), self.wicket_head(last).squeeze(-1)


# --- training --------------------------------------------------------------

def train(format_filter: str | None = "IT20", limit: int | None = None,
          epochs: int = EPOCHS, batch_size: int = BATCH, lr: float = LR,
          device: str | None = None) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        idx = dev.index or 0
        gpu_name = torch.cuda.get_device_name(idx)
        gpu_mem  = torch.cuda.get_device_properties(idx).total_memory // (1 << 20)
        print(f"Device: {dev}  ({gpu_name}, {gpu_mem} MiB)")
    else:
        print(f"Device: {dev}  (CPU — sequence training will be ~20-30x slower than GPU)")

    print(f"Loading features (format={format_filter}, limit={limit}) …")
    df = F.build(format_filter=format_filter, limit=limit)
    if df.empty:
        raise RuntimeError("No rows. Run `pipeline cricsheet` and `pipeline views` first.")
    print(f"  rows: {len(df):,}")

    # 70 / 10 / 20 train / calib / test, all by date
    train_df, holdout_df = F.split_by_date(df, test_frac=0.30)
    calib_df, test_df    = F.split_by_date(holdout_df, test_frac=2 / 3)
    print(f"  train: {len(train_df):,}   calib: {len(calib_df):,}   test: {len(test_df):,}")

    vocab = {
        "batter": _build_vocab(train_df["batter"]),
        "bowler": _build_vocab(train_df["bowler"]),
        "venue":  _build_vocab(train_df["venue"]),
    }
    print(f"  vocab: batters={len(vocab['batter'])}, bowlers={len(vocab['bowler'])}, "
          f"venues={len(vocab['venue'])}")

    train_ds = BallSequenceDataset(train_df, vocab)
    calib_ds = BallSequenceDataset(calib_df, vocab)
    test_ds  = BallSequenceDataset(test_df, vocab)

    collate = lambda b: _collate(b, dev)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    calib_dl = DataLoader(calib_ds, batch_size=batch_size, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, collate_fn=collate)

    model = SequenceBallModel(
        n_numeric=len(F.NUMERIC),
        n_batters=len(vocab["batter"]),
        n_bowlers=len(vocab["bowler"]),
        n_venues=len(vocab["venue"]),
    ).to(dev)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_dl))
    runs_loss = nn.CrossEntropyLoss()
    wkt_loss  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=dev))

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0.0; n = 0
        for batch in train_dl:
            num, bat, bow, ven, mask, yr, yw = batch
            opt.zero_grad()
            r_logits, w_logits = model(num, bat, bow, ven, mask)
            loss = runs_loss(r_logits, yr) + wkt_loss(w_logits, yw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            total += loss.item() * yr.size(0); n += yr.size(0)
        train_loss = total / max(n, 1)

        val_loss = _eval_loss(model, calib_dl, runs_loss, wkt_loss)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            marker = "  ✓"
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"n_numeric": len(F.NUMERIC), "n_batters": len(vocab["batter"]),
                           "n_bowlers": len(vocab["bowler"]), "n_venues": len(vocab["venue"]),
                           "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                           "seq_len": SEQ_LEN},
                "vocab": vocab,
                "numeric_cols": F.NUMERIC,
            }, SEQ_PATH)
        print(f"  epoch {epoch+1:>2}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}{marker}")

    # reload best
    ckpt = torch.load(SEQ_PATH, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # calibrate on calib slice
    print("Fitting calibrators on calib slice …")
    raw_runs_calib, raw_wkt_calib, ya_runs, ya_wkt = _predict_dl(model, calib_dl)
    runs_isos  = C.fit_multiclass(raw_runs_calib, ya_runs, n_classes=len(RUN_BUCKETS))
    wicket_iso = C.fit_binary(raw_wkt_calib, ya_wkt)
    import joblib
    joblib.dump(runs_isos,  SEQ_RUNS_CALIB)
    joblib.dump(wicket_iso, SEQ_WICKET_CALIB)

    # final test metrics (raw + calibrated)
    raw_runs, raw_wkt, ya_runs, ya_wkt = _predict_dl(model, test_dl)
    cal_runs = C.transform_multiclass(runs_isos, raw_runs)
    cal_wkt  = C.transform_binary(wicket_iso, raw_wkt)
    metrics = {
        "rows_train":             int(len(train_df)),
        "rows_calib":             int(len(calib_df)),
        "rows_test":              int(len(test_df)),
        "params":                 sum(p.numel() for p in model.parameters()),
        "runs_logloss_raw":       float(log_loss(ya_runs, raw_runs,
                                                 labels=list(range(len(RUN_BUCKETS))))),
        "runs_logloss_calib":     float(log_loss(ya_runs, cal_runs,
                                                 labels=list(range(len(RUN_BUCKETS))))),
        "runs_top1":              float(accuracy_score(ya_runs, np.argmax(cal_runs, axis=1))),
        "wicket_logloss_raw":     float(log_loss(ya_wkt, raw_wkt)),
        "wicket_logloss_calib":   float(log_loss(ya_wkt, cal_wkt)),
        "wicket_auc":             float(roc_auc_score(ya_wkt, cal_wkt)),
        "format_filter":          format_filter,
        "seq_len":                SEQ_LEN,
    }
    SEQ_META_PATH.write_text(json.dumps(metrics, indent=2))

    print("\n=== sequence-model metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")
    return metrics


def _eval_loss(model, dl, runs_loss_fn, wkt_loss_fn) -> float:
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for batch in dl:
            num, bat, bow, ven, mask, yr, yw = batch
            r_logits, w_logits = model(num, bat, bow, ven, mask)
            loss = runs_loss_fn(r_logits, yr) + wkt_loss_fn(w_logits, yw)
            total += loss.item() * yr.size(0); n += yr.size(0)
    return total / max(n, 1)


def _predict_dl(model, dl):
    runs_p, wkt_p, ys_runs, ys_wkt = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            num, bat, bow, ven, mask, yr, yw = batch
            r_logits, w_logits = model(num, bat, bow, ven, mask)
            r_p = torch.softmax(r_logits, dim=-1).cpu().numpy()
            w_p = torch.sigmoid(w_logits).cpu().numpy()
            runs_p.append(r_p); wkt_p.append(w_p)
            ys_runs.append(yr.cpu().numpy()); ys_wkt.append(yw.cpu().numpy())
    return (np.concatenate(runs_p), np.concatenate(wkt_p),
            np.concatenate(ys_runs), np.concatenate(ys_wkt))


# --- inference --------------------------------------------------------------

_loaded: dict | None = None


def _load(device: str | None = None):
    global _loaded
    if _loaded is not None:
        return _loaded
    if not SEQ_PATH.exists():
        raise RuntimeError("Sequence model not found. Run `pipeline model train --type sequence`.")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(SEQ_PATH, map_location=dev, weights_only=False)
    cfg = ckpt["config"]
    model = SequenceBallModel(
        n_numeric=cfg["n_numeric"], n_batters=cfg["n_batters"],
        n_bowlers=cfg["n_bowlers"], n_venues=cfg["n_venues"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
    ).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    runs_isos = wicket_iso = None
    if SEQ_RUNS_CALIB.exists() and SEQ_WICKET_CALIB.exists():
        import joblib
        runs_isos  = joblib.load(SEQ_RUNS_CALIB)
        wicket_iso = joblib.load(SEQ_WICKET_CALIB)
    _loaded = {"model": model, "vocab": ckpt["vocab"], "device": dev,
               "numeric_cols": ckpt["numeric_cols"],
               "runs_isos": runs_isos, "wicket_iso": wicket_iso}
    return _loaded


def predict_sequence(history: list[dict]) -> dict:
    """Score the *last* state in `history`. Earlier entries provide context.

    `history` is a list of state dicts ordered oldest-first. The last entry
    must be the ball you want to predict. If shorter than SEQ_LEN, the
    sequence is left-padded internally.
    """
    if not history:
        raise ValueError("history must contain at least one ball state")
    L = _load()
    model, vocab, dev = L["model"], L["vocab"], L["device"]
    numeric_cols = L["numeric_cols"]

    seq = history[-SEQ_LEN:]
    pad = SEQ_LEN - len(seq)

    num = np.zeros((SEQ_LEN, len(numeric_cols)), dtype=np.float32)
    bat = np.zeros(SEQ_LEN, dtype=np.int64)
    bow = np.zeros(SEQ_LEN, dtype=np.int64)
    ven = np.zeros(SEQ_LEN, dtype=np.int64)
    mask = np.zeros(SEQ_LEN, dtype=np.float32)

    for j, s in enumerate(seq):
        idx = pad + j
        num[idx] = [float(s.get(c) or 0.0) for c in numeric_cols]
        bat[idx] = _lookup(vocab["batter"], s.get("batter"))
        bow[idx] = _lookup(vocab["bowler"], s.get("bowler"))
        ven[idx] = _lookup(vocab["venue"],  s.get("venue"))
        mask[idx] = 1.0

    with torch.no_grad():
        num_t  = torch.from_numpy(num[None, :, :]).to(dev)
        bat_t  = torch.from_numpy(bat[None, :]).to(dev)
        bow_t  = torch.from_numpy(bow[None, :]).to(dev)
        ven_t  = torch.from_numpy(ven[None, :]).to(dev)
        mask_t = torch.from_numpy(mask[None, :]).to(dev)
        r_logits, w_logits = model(num_t, bat_t, bow_t, ven_t, mask_t)
        rp = torch.softmax(r_logits, dim=-1).cpu().numpy()
        wp = torch.sigmoid(w_logits).cpu().numpy()

    if L["runs_isos"] is not None:
        rp = C.transform_multiclass(L["runs_isos"], rp)
        wp = C.transform_binary(L["wicket_iso"], wp)

    runs_probs = {b: float(rp[0][i]) for i, b in enumerate(RUN_BUCKETS)}
    expected = sum(b * p for b, p in runs_probs.items() if b != 5) + 5 * runs_probs.get(5, 0)
    return {
        "runs_probs":    runs_probs,
        "wicket_prob":   float(wp[0]),
        "expected_runs": expected,
        "calibrated":    L["runs_isos"] is not None,
        "model":         "sequence",
    }

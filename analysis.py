"""Empirical privacy loss of HyperLogLog (p=10) on UCI Online Retail, with and without hash-based subsampling."""
import json, os, socket, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, P, M = 42, 10, 1 << 10
N_SEEDS, N_TUNE_SEEDS = 2000, 1000      # random hash seeds per window
GRID = [0.5, 0.3, 0.2, 0.1, 0.05]        # candidate subsampling rates
MAPE_TOL = 0.05                          # max extra MAPE (absolute) allowed when tuning
socket.setdefaulttimeout(60)
U = np.uint64
SALT, MASK52 = U(0xD6E8FEB86659FD93), U((1 << 52) - 1)


def load():
    from ucimlrepo import fetch_ucirepo
    for k in range(4):                   # retry with backoff
        try:
            data = fetch_ucirepo(id=352).data
            break
        except Exception as e:
            print("download failed:", e); time.sleep(5 * 2 ** k)
    else:
        sys.exit("could not download dataset")
    # InvoiceNo / StockCode have the "ID" role and are not in data.features,
    # so use the full original table (fall back to ids + features).
    X = getattr(data, "original", None)
    if X is None or "InvoiceNo" not in X.columns:
        X = pd.concat([data.ids, data.features], axis=1)
        X = X.loc[:, ~X.columns.duplicated()]
    df = X.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df.sort_values("InvoiceDate").reset_index(drop=True)


def windows(d, min_users=20):
    """Weekly windows -> arrays of distinct CustomerIDs."""
    wk = (d.InvoiceDate - d.InvoiceDate.min()).dt.days // 7
    ws = [np.asarray(a, dtype=np.int64) for a in d.groupby(wk).CustomerID.unique()]
    return [a for a in ws if len(a) >= min_users]


def mix(x):
    """splitmix64 finaliser (vectorised, wraps mod 2^64)."""
    with np.errstate(over="ignore"):
        x = x + U(0x9E3779B97F4A7C15)
        x = (x ^ (x >> U(30))) * U(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> U(27))) * U(0x94D049BB133111EB)
        return x ^ (x >> U(31))


def sketch(items, seeds, q):
    """HLL registers (n_seeds, M); items kept iff an independent hash < q (q=1: plain HLL)."""
    h = mix(mix(seeds)[:, None] ^ items.astype(np.uint64)[None, :])
    keep = (mix(h ^ SALT) >> U(11)).astype(np.float64) / 2.0 ** 53 < q
    idx = (h >> U(64 - P)).astype(np.int64)
    _, e = np.frexp(((h >> U(2)) & MASK52).astype(np.float64))
    rank = np.where(keep, 53 - e, 0)     # rank = leading zeros in 52 bits + 1
    n = len(seeds)
    regs = np.zeros(n * M, dtype=np.int64)
    np.maximum.at(regs, (np.arange(n)[:, None] * M + idx).ravel(), rank.ravel())
    return regs.reshape(n, M)


def estimate(regs, q):
    """Standard HLL estimator with linear-counting small-range correction, rescaled by 1/q."""
    alpha = 0.7213 / (1 + 1.079 / M)
    raw = alpha * M * M / np.sum(2.0 ** -regs.astype(float), axis=1)
    zeros = (regs == 0).sum(1)
    lc = M * np.log(M / np.maximum(zeros, 1))
    return np.where((raw <= 2.5 * M) & (zeros > 0), lc, raw) / q


def emp_epsilon(a, b, nbins=8, min_count=30):
    """Max |log P_a(bin)/P_b(bin)| over pooled-quantile bins with >= min_count samples in both."""
    edges = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0, 1, nbins + 1))[1:-1])
    ca = np.bincount(np.digitize(a, edges), minlength=len(edges) + 1)
    cb = np.bincount(np.digitize(b, edges), minlength=len(edges) + 1)
    ok = (ca >= min_count) & (cb >= min_count)
    return float(np.max(np.abs(np.log(ca[ok] / cb[ok])))) if ok.any() else 0.0


def evaluate(wins, q, seeds):
    """Per window: empirical epsilon (stream vs stream + one new user) and MAPE of the count."""
    eps, mape = [], []
    for users in wins:
        regs = sketch(users, seeds, q)
        extra = sketch(np.array([users.max() + 1]), seeds, q)   # one added, previously unseen user
        a, b = estimate(regs, q), estimate(np.maximum(regs, extra), q)
        eps.append(emp_epsilon(a, b))
        mape.append(np.mean(np.abs(a - len(users)) / len(users)))
    return np.array(eps), np.array(mape)


def main():
    df = load()
    inv = df.groupby("InvoiceNo")["InvoiceDate"].min().sort_values()
    cut = inv.iloc[int(0.7 * len(inv))]                       # time-ordered split by invoice
    is_tr = df.groupby("InvoiceNo")["InvoiceDate"].transform("min") < cut
    train, test = df[is_tr], df[~is_tr]
    wtr, wte = windows(train), windows(test)
    print(f"rows={len(df)} train={len(train)} test={len(test)} windows train/test={len(wtr)}/{len(wte)}")
    if len(wtr) < 5 or len(wte) < 5:
        sys.exit("too few windows")

    # Tuning on training windows only (blocked CV: each weekly window is a fold)
    seeds = np.random.default_rng(SEED).integers(0, 2 ** 63, N_TUNE_SEEDS, dtype=np.uint64)
    _, base_mape = evaluate(wtr, 1.0, seeds)
    cand = {}
    for q in GRID:
        e, m = evaluate(wtr, q, seeds)
        cand[q] = (e.mean(), m.mean())
        print(f"  tune q={q}: eps={e.mean():.3f} mape={m.mean():.3f} (baseline mape={base_mape.mean():.3f})")
    ok = [q for q in GRID if cand[q][1] <= base_mape.mean() + MAPE_TOL]
    q_star = min(ok, key=lambda q: cand[q][0]) if ok else min(GRID, key=lambda q: cand[q][1])
    print("chosen rate:", q_star)

    # Single final evaluation on held-out test windows, 3 runs with different hash-seed sets
    res = {"base": [], "imp": []}
    for run in (42, 43, 44):
        s = np.random.default_rng(run).integers(0, 2 ** 63, N_SEEDS, dtype=np.uint64)
        for k, q in (("base", 1.0), ("imp", q_star)):
            e, m = evaluate(wte, q, s)
            res[k].append((e.mean(), m.mean(), e.max()))
        print(f"run {run}: base={res['base'][-1]} imp={res['imp'][-1]}")
    r = {k: np.array(v) for k, v in res.items()}
    metrics = {"baseline_score": r["base"][:, 0].mean(), "improved_score": r["imp"][:, 0].mean(),
               "baseline_score_std": r["base"][:, 0].std(), "improved_score_std": r["imp"][:, 0].std(),
               "baseline_mape": r["base"][:, 1].mean(), "improved_mape": r["imp"][:, 1].mean(),
               "baseline_mape_std": r["base"][:, 1].std(), "improved_mape_std": r["imp"][:, 1].std(),
               "baseline_worst_window_epsilon": r["base"][:, 2].mean(),
               "improved_worst_window_epsilon": r["imp"][:, 2].mean(),
               "chosen_sampling_rate": q_star, "n_train_windows": len(wtr), "n_test_windows": len(wte)}
    metrics["improvement"] = metrics["baseline_score"] - metrics["improved_score"]
    metrics["mape_increase"] = metrics["improved_mape"] - metrics["baseline_mape"]
    metrics = {k: float(v) for k, v in metrics.items()}
    if not all(np.isfinite(list(metrics.values()))):
        sys.exit("non-finite metrics")

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    for a, key, lab in ((ax[0], "score", "mean empirical epsilon"), (ax[1], "mape", "MAPE of distinct count")):
        a.bar(["HLL", f"HLL + subsample q={q_star}"], [metrics[f"baseline_{key}"], metrics[f"improved_{key}"]],
              yerr=[metrics[f"baseline_{key}_std"], metrics[f"improved_{key}_std"]], color=["gray", "tab:blue"])
        a.set_ylabel(lab)
    plt.tight_layout(); plt.savefig("figures/baseline_vs_improved.png", dpi=120)

    out = {"question": "On a real transaction log, how large is the empirical privacy loss of a standard HyperLogLog sketch, and can hash-based subsampling cut it at a small cost in accuracy?",
           "method": "Weekly windows of distinct CustomerIDs from UCI Online Retail; time split 70/30 by invoice; subsampling rate tuned on training windows; per test window, outputs over random hash seeds for stream vs stream plus one new user are binned by quantiles and epsilon is the max abs log-probability ratio over bins with >=30 samples in both.",
           "dataset": {"name": "UCI Online Retail (rows with CustomerID)", "source_url": "https://archive.ics.uci.edu/dataset/352/online+retail",
                       "n_rows": len(df), "n_train": len(train), "n_test": len(test),
                       "same_as_paper": False, "baseline_is_papers_method": False},
           "metric_name": "mean empirical epsilon over held-out weekly windows (lower is better)", "higher_is_better": False,
           "baseline": "Standard HyperLogLog, precision 10, random seeded 64-bit hash, bias-corrected (linear counting) estimator.",
           "improved": "Same HyperLogLog preceded by hash-based Bernoulli subsampling of items at a rate tuned on the training windows; count rescaled by 1/rate.",
           "metrics": metrics,
           "notes": "Empirical epsilon is a statistical lower-bound-style estimate from 2000 hash seeds per window and 8 quantile bins; it has a positive noise floor even for identical distributions and depends on binning and the min-count threshold. Only one added user is tested per window. Windows are weekly (some partial) with >=20 customers. Selection rule: lowest training epsilon with MAPE at most 5 points above baseline. Mean/std are over three hash-seed sets (42-44); data is not resampled."}
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

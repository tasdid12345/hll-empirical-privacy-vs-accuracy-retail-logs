# HyperLogLog empirical privacy vs accuracy on UCI Online Retail

Measures the empirical privacy loss (epsilon) of a standard HyperLogLog sketch (precision 10) on weekly windows of distinct CustomerIDs, and tests whether hash-based subsampling (rate tuned on training windows only) reduces it at a small accuracy cost.

Run: `pip install -r requirements.txt && python analysis.py`

Outputs `results.json` and `figures/baseline_vs_improved.png`.

Fix note: `InvoiceNo` has the ID role in ucimlrepo and is not in `data.features`; the script now uses the full original table. The method is unchanged.

## Results (latest run)

On the UCI Online Retail CustomerID log (11 held-out weekly windows), a standard HyperLogLog with a random seeded 64-bit hash had mean empirical epsilon 0.172, versus 0.086 with hash-based subsampling at rate 0.5, though count MAPE rose from 1.9% to 4.1%.

| Metric | Value |
|---|---|
| baseline_score | 0.1716 |
| improved_score | 0.08558 |
| baseline_score_std | 0.004026 |
| improved_score_std | 0.002569 |
| baseline_mape | 0.01908 |
| improved_mape | 0.04114 |
| baseline_mape_std | 0.000104 |
| improved_mape_std | 0.0001816 |
| baseline_worst_window_epsilon | 0.2311 |
| improved_worst_window_epsilon | 0.1326 |
| chosen_sampling_rate | 0.5 |
| n_train_windows | 43 |
| n_test_windows | 11 |
| improvement | 0.08599 |
| mape_increase | 0.02206 |

Run on 2026-10-30. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Empirical epsilon is a statistical lower-bound-style estimate from 2000 hash seeds per window and 8 quantile bins; it has a positive noise floor even for identical distributions and depends on binning and the min-count threshold. Only one added user is tested per window. Windows are weekly (some partial) with >=20 customers. Selection rule: lowest training epsilon with MAPE at most 5 points above baseline. Mean/std are over three hash-seed sets (42-44); data is not resampled.

## Reference

This project is motivated by the following paper:

> C Dickens, J Thaler, D Ting (2022). *Order-invariant cardinality estimators are differentially private*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/623307df18da128262aaf394cdcfb235-Abstract-Conference.html

Paper author: Prof. Justin Thaler, Georgetown University (justin.thaler@georgetown.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

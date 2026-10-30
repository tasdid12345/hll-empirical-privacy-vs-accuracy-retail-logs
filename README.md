# Empirical privacy loss of HyperLogLog on a real transaction log

**Motivation.** The paper "Order-invariant cardinality estimators are differentially private" (2022) shows that order-invariant sketches such as HyperLogLog are differentially private under a random hash, and that subsampling items can improve the privacy guarantee. This small stand-in experiment measures the *empirical* privacy loss of a standard HyperLogLog on a real stream. It is not a re-implementation of the paper.

**Question.** On a real transaction log, how large is the empirical privacy loss of a standard HyperLogLog distinct-count sketch, and can hash-based subsampling cut it at a small cost in accuracy?

**Dataset.** UCI Online Retail (ucimlrepo id 352), using distinct CustomerIDs per weekly window as the stream. This is not the paper's own dataset. Invoices are split by time: the first 70% are used to tune the subsampling rate, and the last 30% form the held-out weekly test windows.

**Baseline vs improved.**
- Baseline: standard HyperLogLog (precision 10, random seeded hash, bias-corrected estimator).
- Improved: the same HyperLogLog preceded by hash-based Bernoulli subsampling of items, with a rate chosen on the training windows and the count rescaled.

For each window, the streams with and without one added user are compared over 1,000+ random hash seeds. The empirical epsilon is the largest log-probability ratio across quantile bins with enough samples. Secondary metrics are the MAPE of the distinct count and the worst-window epsilon. Everything is run with seeds 42, 43 and 44.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> C Dickens, J Thaler, D Ting (2022). *Order-invariant cardinality estimators are differentially private*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/623307df18da128262aaf394cdcfb235-Abstract-Conference.html

Paper author: Prof. Justin Thaler, Georgetown University (justin.thaler@georgetown.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

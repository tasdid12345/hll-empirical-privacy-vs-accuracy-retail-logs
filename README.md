# HyperLogLog empirical privacy vs accuracy on UCI Online Retail

Measures the empirical privacy loss (epsilon) of a standard HyperLogLog sketch (precision 10) on weekly windows of distinct CustomerIDs, and tests whether hash-based subsampling (rate tuned on training windows only) reduces it at a small accuracy cost.

Run: `pip install -r requirements.txt && python analysis.py`

Outputs `results.json` and `figures/baseline_vs_improved.png`.

Fix note: `InvoiceNo` has the ID role in ucimlrepo and is not in `data.features`; the script now uses the full original table. The method is unchanged.

## Reference

This project is motivated by the following paper:

> C Dickens, J Thaler, D Ting (2022). *Order-invariant cardinality estimators are differentially private*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/623307df18da128262aaf394cdcfb235-Abstract-Conference.html

Paper author: Prof. Justin Thaler, Georgetown University (justin.thaler@georgetown.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

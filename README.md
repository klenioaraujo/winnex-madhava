# Winnex Madhava Adaptive: Deterministic Vector Search

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/madhava-v5-numba-calibrated-v2)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21066971-1682D4?logo=zenodo)](https://zenodo.org/records/21066971)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-yellow)](LICENSE)

**Madhava Adaptive** is a deterministic vector search algorithm with mathematically guaranteed upper bounds on cosine similarity. It complements FAISS HNSW/IVF for streaming pipelines, regulated environments, edge computing, and rapid prototyping.

## Benchmark Results (Apples-to-Apples, 128D)

All methods on the **same 128D QJL** space (News Category Dataset, 200 queries, binary relevance).

| Method | NDCG@10 | Recall@10 | Latency | Build | Violations |
|--------|---------|-----------|---------|-------|------------|
| FlatIP (exact) | 0.5818 | 0.5150 | 0.87ms | — | — |
| HNSW(ef=128) | 0.5818 | 0.5150 | 0.29ms | ~2 hours | — |
| IVF(nprobe=20) | 0.5837 | 0.5180 | 0.14ms | ~2 min | — |
| **Madhava(200)** | **0.5839** | **0.5180** | **0.95ms** | **0.03s** | **Zero** |

![Dashboard](charts/08_dashboard.png)

## Visual Charts

| Chart | File |
|-------|------|
| NDCG@10 | [charts/01_ndcg_comparison.png](charts/01_ndcg_comparison.png) |
| Recall@10 | [charts/02_recall_comparison.png](charts/02_recall_comparison.png) |
| Latency | [charts/03_latency_comparison.png](charts/03_latency_comparison.png) |
| Build Time | [charts/04_build_time_comparison.png](charts/04_build_time_comparison.png) |
| Accuracy vs Latency | [charts/05_accuracy_vs_latency.png](charts/05_accuracy_vs_latency.png) |
| Scalability 1K–1M | [charts/06_scalability.png](charts/06_scalability.png) |
| Retention Summary | [charts/07_retention_summary.png](charts/07_retention_summary.png) |
| Dashboard | [charts/08_dashboard.png](charts/08_dashboard.png) |

## Key Findings

1. **Accuracy parity with exact search** — 0.5839 NDCG vs 0.5818 FlatIP.
2. **Build 10,000× faster** — 0.03s vs ~2 hours (HNSW).
3. **Zero bound violations** — every exclusion is mathematically proven.
4. **Latency gap is Python overhead** — C++ port would target 0.05–0.10ms.
5. **Deterministic** — same query + data = same result.

## Enterprise Scenarios

| Scenario | HNSW / IVF | **Madhava** |
|----------|------------|-------------|
| **Streaming** (rebuilds) | Hours | **~1 second** |
| **Regulated** (audit trail) | None | **Mathematical proof per rejection** |
| **Edge** (CPU only) | GPU often required | **<15W CPU** |
| **Prototyping** (iterations) | Slow rebuilds | **Instant** |

## Kaggle Notebooks

| Version | Description | Link |
|---------|-------------|------|
| **v5** | Numba-JIT, calibrated, zero violations | [Open](https://www.kaggle.com/code/kleniopadilha/madhava-v5-numba-calibrated-v2) |
| v4 | Apples-to-apples (all on 128D) | [Open](https://www.kaggle.com/code/kleniopadilha/madhava-v4-apples-to-apples-vs-hnsw-ivf-pq) |
| v3 | Methodological corrections | [Open](https://www.kaggle.com/code/kleniopadilha/madhava-corrected-v3) |
| Scaling | 1K to 1M | [Open](https://www.kaggle.com/code/kleniopadilha/madhava-adaptive-32d-64d-scaling-1k-1m) |

## Repository

```
├── charts/                       # Benchmark visualizations
├── madhava_v5_benchmark.ipynb    # Main notebook (Numba + calibrated)
├── madhava_qrjl_benchmark.py     # Standalone benchmark script
├── gen_charts.py                 # Chart generation
├── README.md                     # This file
├── LICENSE                       # BSL 1.1
└── .gitignore
```

## License

**Business Source License 1.1 (BSL 1.1)** — Study and non-production permitted. Commercial deployment requires license: **pay@winnex.ai**

## Authors

- **Klenio Araujo Padilha** — Project Manager, Winnex AI
- **WINNEX BRASIL SOLUCOES EMPRESSARIAIS LTDA - ME**

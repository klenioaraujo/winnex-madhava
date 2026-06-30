# Madhava v12 Configuration Guide

The entire Madhava stack is controlled by `config.json`. This document explains every parameter.

## Quick Start

```bash
# Default config ([32,64] stage dimensions)
python madhava_v12.py

# Use [64,128] stage dimensions
python madhava_v12.py --config config_64_128.json

# Quick test (50K vectors, 200 queries)
python madhava_v12.py --quick

# As a module
from madhava_v12 import MadhavaCore, Madhybrid, load_config
cfg = load_config('config.json)
mc = MadhavaCore(cfg)
mc.build(vectors)
results = mc.search(query)
```

## Configuration Parameters

### `dimensions` — Search Space Dimensionality

```json
{
  "dimensions": {
    "input_dim": 128,
    "stage_dims": [32, 64],
    "qjl_dim": 128
  }
}
```

| Parameter | Description | Values |
|-----------|-------------|--------|
| `input_dim` | Dimensionality of the input vectors | 128 (SIFT), 384 (SBERT) |
| `stage_dims` | Progressive refinement pipeline | `[32,64]` (fast), `[64,128]` (precise), `[16,32,64,128]` (multi-stage) |
| `qjl_dim` | Johnson-Lindenstrauss target dimension | 128 (standard), ≤ input_dim |

**Tip**: `[64,128]` achieves R@10=1.000 (matches exact search) on SIFT-1M but is ~2x slower than `[32,64]`.

### `search` — Adaptive Keep-Ratio and Refinement

```json
{
  "search": {
    "adaptive_keep_base": 0.25,
    "adaptive_keep_min": 0.05,
    "adaptive_keep_max": 0.50,
    "adaptive_bounds_sensitivity": 0.12,
    "stage2_topk": 500,
    "stage2_topk_max": 2000,
    "final_results": 10,
    "epsilon": 1e-5
  }
}
```

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `adaptive_keep_base` | Base ratio for Stage 1 retention | 0.25 | 0.05 — 0.50 |
| `adaptive_keep_min` | Minimum retention ratio | 0.05 | 0.01 — 0.20 |
| `adaptive_keep_max` | Maximum retention ratio | 0.50 | 0.10 — 1.00 |
| `adaptive_bounds_sensitivity` | Sensitivity of keep ratio to bound range | 0.12 | 0.01 — 1.00 |
| `stage2_topk` | Number of candidates at stage 2 | 500 | 100 — 2000 |
| `stage2_topk_max` | Hard cap for stage 2 | 2000 | — |
| `final_results` | Number of results returned | 10 | 1 — 100 |
| `epsilon` | Small constant for floating-point safety | 1e-5 | 1e-9 — 1e-3 |

The adaptive keep ratio formula:
```
raw_keep = adaptive_keep_base * adaptive_bounds_sensitivity / max(bound_range, 0.01)
keep = clip(raw_keep, adaptive_keep_min, adaptive_keep_max)
```

**For structured data** (SBERT, GloVe): bound range is large → lower keep → faster.
**For uniform data** (random, SIFT): bound range is narrow → higher keep → more accurate.

### `hybrid` — IVF Clustering + Madhava per Cell

```json
{
  "hybrid": {
    "enabled": true,
    "n_cells": 64,
    "n_probe": [3, 5, 8, 10, 15],
    "clustering": {
      "algorithm": "MiniBatchKMeans",
      "random_state": 42,
      "batch_size": 20000,
      "n_init": 3,
      "max_iter": 50
    }
  }
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable MadHybrid clustering | true |
| `n_cells` | Number of Voronoi cells | 64 |
| `n_probe` | List of probe values for benchmarking | [3,5,8,10,15] |

**Larger n_cells** → finer partitioning → higher recall at low n_probe. Memory scales with O(n_cells × stage_dims × cell_size).

### `bounds` — Mathematical Guarantees

```json
{
  "bounds": {
    "cauchy_schwarz_epsilon": 1e-5,
    "orthogonality_tolerance": 1e-5,
    "error_clip_min": 0.0
  }
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cauchy_schwarz_epsilon` | Epsilon added to upper bound (float safety) | 1e-5 |
| `orthogonality_tolerance` | Max deviation from identity for QR matrix | 1e-5 |

These ensure the mathematical guarantee holds. Do not change unless you understand the implications.

### `modulation` — Error Backpropagation

```json
{
  "modulation": {
    "error_backprop": true,
    "alpha_smoothing": 0.5,
    "alpha_min": 0.01,
    "alpha_max": 0.99
  }
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `error_backprop` | Enable B1 + alpha*(B2-B1) score blending | true |
| `alpha_smoothing` | Sigmoid steepness for alpha computation | 0.5 |

### `streaming` — Real-Time Index Configuration

```json
{
  "streaming": {
    "max_rebuild_frequency_ms": 5000,
    "dynamic_insert_batch": 1000,
    "cache_projections": true
  }
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_rebuild_frequency_ms` | Minimum time between rebuilds (ms) | 5000 |
| `dynamic_insert_batch` | Batch size for incremental inserts | 1000 |
| `cache_projections` | Cache pre-computed projections in RAM | true |

### `benchmark` — Test Configuration

```json
{
  "benchmark": {
    "dataset": "sift-1m",
    "n_queries": 500,
    "recall_k": 10,
    "ground_truth_method": "flatip"
  }
}
```

## Configuration Presets

| File | stage_dims | Use Case |
|------|-----------|----------|
| `config.json` | [32, 64] | Default: balanced speed/recall |
| `config_64_128.json` | [64, 128] | High precision: matches exact search |
| `config_streaming.json` | [32, 64] | Optimized for frequent rebuilds |

## Programmatic Usage

```python
from madhava_v12 import MadhavaCore, Madhybrid, load_config
import numpy as np

# Load configuration
cfg = load_config('config.json')

# Standalone search (1 cell over full corpus)
mc = MadhavaCore(cfg)
mc.build(vectors)  # vectors: np.ndarray of shape (N, D)
results = mc.search(query, k=10)  # returns top-k indices

# Hybrid search (clustered + Madhava per cell)
mh = Madhybrid(cfg)
mh.build(vectors)
results = mh.search(query, k=10, n_probe=5)

# Dynamic reconfiguration
cfg['dimensions']['stage_dims'] = [64, 128]
cfg['search']['stage2_topk'] = 1000
mc2 = MadhavaCore(cfg)
```

## Performance Tuning

**For maximum recall**: `stage_dims=[64,128]`, `stage2_topk=1000`
**For maximum speed**: `stage_dims=[32,64]`, `adaptive_keep_max=0.10`
**For streaming**: keep `n_cells=64`, set `adaptive_keep_min=0.10`
**For regulated**: keep defaults (all guarantees active)

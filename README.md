# Winnex MadHybrid v11: Real SIFT-1M Corrected Benchmark

## Corrected Metrics

Recall@10 is now computed against **FlatIP exact search** as ground truth.
This eliminates the 'super-exact' illusion from prior versions.

## Results (SIFT-1M, 100K sample, 500 queries)

| Method | R@10 | Lat(ms) | QPS | Build | Bound |
|---|---|---|---|---|---|
| HNSW(ef=256) | 0.9994 | 0.23ms | 4363 | 1.4s | None |
| IVF(nprobe=20) | 0.9824 | 0.12ms | 8350 | <1m | None |
| **MadHybrid(np=15)** | **0.9930** | **2.04ms** | **489** | **5.0s** | **Zero** |
| MadHybrid(np=10) | 0.9818 | 1.38ms | 725 | 5.0s | Zero |
| MadHybrid(np=8) | 0.9688 | 1.12ms | 895 | 5.0s | Zero |

## Streaming Niche Positioning

MadHybrid is **not a HNSW substitute**. The streaming use case is:

- **Rebuild every 1-60 seconds** (12 rebuilds/minute at 5s each)
- **Zero bound violations** mathematically guaranteed
- **Deterministic**: same query + data = same result, always
- **CPU-only**: no GPU required for indexing or inference

## References

GitHub: https://github.com/klenioaraujo/winnex-madhava
Zenodo: https://zenodo.org/records/21052709

License: BSL 1.1 | pay@winnex.ai

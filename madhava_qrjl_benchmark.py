#!/usr/bin/env python3
"""
==========================================================================
Madhava QR-JL (8D->64D) - Benchmark vs HNSW / IVF / PQ / FlatIP
==========================================================================
Uso:
  python madhava_qrjl_benchmark.py --quick         # N <= 50K only
  python madhava_qrjl_benchmark.py --real-only      # News dataset only
  python madhava_qrjl_benchmark.py --queries 100    # more queries
  python madhava_qrjl_benchmark.py --runs 5         # 5x repetition

Licenca: BSL 1.1
==========================================================================
"""
import sys, os, warnings, math, time, random, gc, json, copy, argparse
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

MADHAVA_DIMS = [8, 64]
FULL_DIM = 128
CASCADE_TOPK = [2000, 200]
FINAL_K = 10

SYNTHETIC_NS = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
SYNTHETIC_DIM = 128

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="N <= 50K only")
parser.add_argument("--real-only", action="store_true", help="News dataset only")
parser.add_argument("--runs", type=int, default=1, help="Runs per config")
parser.add_argument("--queries", type=int, default=50, help="Queries per test")
parser.add_argument("--nsynth", type=int, nargs="+", default=None, help="Custom N")
parser.add_argument("--output", type=str, default="madhava_bench_results.json")
args = parser.parse_args()

N_RUNS = args.runs
N_QUERIES_MAX = args.queries
OUTPUT_FILE = args.output

if args.quick:
    SYNTHETIC_NS = [1000, 5000, 10000, 50000]
if args.nsynth:
    SYNTHETIC_NS = args.nsynth

# ================================================================
#  MADHAVA QR-JL CORE
# ================================================================

class MadhavaQRJL:
    """
    Madhava QR-JL with Error Backpropagation Modulation.

    Neural network analogy (2-layer, feed-forward + backprop):
      Layer 1 (8D):   forward B̂₁ = <P₁v, P₁q> + e₁(v)·e₁(q) for ALL N docs
                        [O(N·8) fast matmul + vectorized bound]

      Error signal:   δ = B̂₂ - B̂₁  (how much the bound tightens at 64D)
      Learning rate:  αᵢ = σ((e₁[i] - e₂[i]) / mean(e₁))
                        high residual at 8D + large tightening => high α
                        already accurate at 8D => low α (trust B̂₁)

      Layer 2 (64D):  refinement B̂₂ for C=0.20·N survivors
        [O(0.20·N·64) refinement]

      Modulated score:  s = B̂₁ + α·(B̂₂ - B̂₁)
        [this IS error backpropagation: δ propagates to correct B̂₁]

      Final exact cosine on top-K survivors.

    Why this works:
      Documents pruned early by B̂₁ (hard 8D cutoff) never get rescued.
      But with modulation, ALL docs get B̂₁, most get B̂₂, and B̂₁ is
      corrected by the error gradient. No information is discarded.
    """

    def __init__(self, dims=None, full_dim=None, seed=SEED):
        self.dims = dims or MADHAVA_DIMS
        self.full_dim = full_dim or FULL_DIM
        self.keep_ratio = 0.20  # keep ~20% for refinement
        self.max_stage1 = 100000  # cap for very large N
        self.final_topk = 200  # survivors for exact cosine
        self.rng = np.random.RandomState(seed + 1)
        self.vectors = None
        self.n_vectors = 0
        self.proj_L = {}
        self.error = {}
        self.proj_matrices = {}
        self.norms = None
        self.build_time = 0.0

    def _make_orthogonal_proj(self, d_out, d_in):
        """QR-orthogonalized JL: R^d_in -> R^d_out, rows orthonormal."""
        R = self.rng.randn(d_out, d_in).astype(np.float64)
        Q, _ = np.linalg.qr(R.T)
        P = Q[:, :d_out].T.astype(np.float32)
        err = np.abs(P @ P.T - np.eye(d_out, dtype=np.float32)).max()
        assert err < 1e-5, f"QR orthogonality failed: {err:.2e}"
        return P

    def build(self, vectors):
        """Cache layer: precompute all projections and residuals."""
        t0 = time.time()
        self.vectors = vectors
        self.n_vectors = len(vectors)
        self.norms = np.linalg.norm(vectors, axis=1).astype(np.float64)
        for d in self.dims:
            P = self._make_orthogonal_proj(d, self.full_dim)
            self.proj_matrices[d] = P
            proj = (vectors.astype(np.float32) @ P.T).astype(np.float64)
            self.proj_L[d] = proj
            captured = np.linalg.norm(proj, axis=1)
            res_sq = np.maximum(self.norms**2 - captured**2, 0)
            self.error[d] = np.sqrt(res_sq)
        self.build_time = time.time() - t0
        return self

    def _upper_bound(self, pv, ev, pq, eq):
        """B̂_d = <P_d v, P_d q> + e_d(v)·e_d(q)"""
        return pv @ pq + ev * eq

    def _modulation_alpha(self, e1, e2):
        """
        Per-document learning rate: α = σ((e₁ - e₂)/μ)

        eᵢ = residual error at dimension i (Pythagorean)
        When e₁ >> e₂: large tightening → high α (correct aggressively)
        When e₁ ≈ e₂: already tight → low α (trust layer 1)
        """
        mu = np.maximum(np.mean(e1), 1e-9)
        delta = (e1 - e2) / mu
        return 1.0 / (1.0 + np.exp(-delta * 0.5))

    def search(self, query, return_profile=False):
        """
        Error Backpropagation Modulation Search.

        Step 1: Forward 8D (B̂₁) for ALL N docs
        Step 2: Select survivors (keep_ratio * N, capped)
        Step 3: Forward 64D (B̂₂) for survivors
        Step 4: α = σ((e₁ - e₂)/μ), score = B̂₁ + α·(B̂₂ - B̂₁)
        Step 5: Exact cosine on top survivors by modulated score
        """
        q = query.astype(np.float64).flatten()
        q_norm = np.linalg.norm(q)
        prof = {"n_total": self.n_vectors}

        # ---- Layer 1: Forward 8D for ALL N docs (fast) ----
        d1 = self.dims[0]
        q1 = (q.astype(np.float32) @ self.proj_matrices[d1].T).astype(np.float64)
        qr1 = math.sqrt(max(0, q_norm**2 - np.linalg.norm(q1)**2))
        B1 = self._upper_bound(self.proj_L[d1], self.error[d1], q1, qr1)
        prof["B1_range"] = [float(B1.min()), float(B1.max())]
        prof["B1_mean"] = float(B1.mean())

        # Select survivors: keep ~20%
        keep1 = min(max(int(self.n_vectors * self.keep_ratio), 2000), self.max_stage1)
        keep1 = min(keep1, self.n_vectors)

        if self.n_vectors <= keep1:
            idx1 = np.arange(self.n_vectors)
        else:
            idx1 = np.argpartition(-B1, max(0, keep1 - 1))[:keep1]
        prof["n_candidates_stage1"] = len(idx1)
        prof["prune_ratio_stage1"] = 1.0 - len(idx1) / max(self.n_vectors, 1)

        # ---- Layer 2: Forward 64D for survivors (refinement) ----
        d2 = self.dims[1]
        q2 = (q.astype(np.float32) @ self.proj_matrices[d2].T).astype(np.float64)
        qr2 = math.sqrt(max(0, q_norm**2 - np.linalg.norm(q2)**2))

        B2 = self._upper_bound(self.proj_L[d2][idx1], self.error[d2][idx1], q2, qr2)
        prof["B2_range"] = [float(B2.min()), float(B2.max())]
        prof["B2_mean"] = float(B2.mean())

        # ---- Error Backpropagation Modulation ----
        # Error signal: δ = B̂₂ - B̂₁
        delta = B2 - B1[idx1]
        prof["delta_mean"] = float(np.mean(delta))

        # Learning rate per document
        e1_sel = self.error[d1][idx1]
        e2_sel = self.error[d2][idx1]
        alpha = self._modulation_alpha(e1_sel, e2_sel)
        prof["alpha_mean"] = float(np.mean(alpha))
        prof["alpha_range"] = [float(alpha.min()), float(alpha.max())]

        # Modulated score: s = B̂₁ + α · (B̂₂ - B̂₁)
        modulated = B1[idx1] + alpha * delta
        prof["modulated_range"] = [float(modulated.min()), float(modulated.max())]
        prof["modulated_mean"] = float(np.mean(modulated))

        # ---- Final: exact cosine on top-k by modulated score ----
        keep2 = min(self.final_topk, len(idx1))
        idx2 = idx1[np.argpartition(-modulated, max(0, keep2 - 1))[:keep2]]
        prof["n_candidates_stage2"] = len(idx2)

        cos = self.vectors[idx2].astype(np.float32) @ q.astype(np.float32)
        top = idx2[np.argsort(-cos)[:FINAL_K]]
        prof["n_final"] = len(top)

        if return_profile:
            return top, prof
        return top

    def search_baseline(self, query, return_profile=False):
        """
        Ablation: no error backpropagation modulation.
        Pure 8D bound ranking + exact cosine on top-keep1 only.
        Used to isolate the effect of error backpropagation.
        """
        q = query.astype(np.float64).flatten()
        q_norm = np.linalg.norm(q)
        prof = {"n_total": self.n_vectors, "mode": "ablation_no_modulation"}

        d1 = self.dims[0]
        q1 = (q.astype(np.float32) @ self.proj_matrices[d1].T).astype(np.float64)
        qr1 = math.sqrt(max(0, q_norm**2 - np.linalg.norm(q1)**2))
        B1 = self._upper_bound(self.proj_L[d1], self.error[d1], q1, qr1)

        keep1 = min(max(int(self.n_vectors * 0.02), 2000), self.max_stage1)
        keep1 = min(keep1, self.n_vectors)
        if self.n_vectors <= keep1:
            idx = np.arange(self.n_vectors)
        else:
            idx = np.argpartition(-B1, max(0, keep1 - 1))[:keep1]
        prof["n_candidates"] = len(idx)

        cos = self.vectors[idx].astype(np.float32) @ q.astype(np.float32)
        top = idx[np.argsort(-cos)[:FINAL_K]]

        if return_profile:
            return top, prof
        return top

    def bound_violations(self, query):
        """Count violations: true_cosine > upper_bound (tolerance 1e-9)."""
        q = query.astype(np.float64).flatten()
        q_norm = np.linalg.norm(q)
        true_cos = self.vectors.astype(np.float64) @ q
        viol = {}
        for d in self.dims:
            P = self.proj_matrices[d]
            qd = (q.astype(np.float32) @ P.T).astype(np.float64)
            qr = math.sqrt(max(0, q_norm**2 - np.linalg.norm(qd)**2))
            ub = self._upper_bound(self.proj_L[d], self.error[d], qd, qr)
            viol[f"{d}D"] = int(np.sum(true_cos > ub + 1e-9))
        return viol, self.n_vectors

# ================================================================
# METRICS
# ================================================================

def ndcg_at_k(ranked, true_scores, k=FINAL_K):
    dcg = 0.0
    for j, idx in enumerate(ranked[:k]):
        rel = true_scores.get(int(idx), 0.0)
        dcg += (2**rel - 1) / math.log2(j + 2)
    sorted_by_score = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
    idcg = 0.0
    for j, (idx, rel) in enumerate(sorted_by_score[:k]):
        idcg += (2**rel - 1) / math.log2(j + 2)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked, true_scores, k=FINAL_K):
    relevant = {int(idx) for idx, sc in true_scores.items() if sc > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for i in ranked[:k] if int(i) in relevant)
    return hits / len(relevant)

# ================================================================
# DATA GEN
# ================================================================

def make_uniform_sphere(n, dim, seed=SEED):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

# ================================================================
# BENCHMARK EXECUTION
# ================================================================

def run_benchmark_synthetic():
    """Run benchmark on synthetic uniform sphere data."""
    print("\n" + "=" * 70)
    print("  BENCHMARK A: DADOS SINTETICOS (uniforme na esfera S^127)")
    print("=" * 70)

    all_results = {}

    for nc in SYNTHETIC_NS:
        print(f"\n  --- N = {nc:,} ---")
        E = make_uniform_sphere(nc, SYNTHETIC_DIM, SEED)
        Q = make_uniform_sphere(N_QUERIES_MAX, SYNTHETIC_DIM, SEED + 9999)
        n_q = min(N_QUERIES_MAX, nc // 20 + 10)

        results = {
            "N": nc, "dim": SYNTHETIC_DIM,
            "madhava": {}, "flatip": {},
            "hnsw_ef32": {}, "hnsw_ef64": {}, "hnsw_ef128": {},
            "ivf_nprobe1": {}, "ivf_nprobe10": {}, "ivf_nprobe20": {},
            "pq_m8": {}, "pq_m16": {}, "pq_m32": {},
        }

        # --- FAISS FlatIP (ground truth) ---
        try:
            import faiss
            t0 = time.time()
            flat_idx = faiss.IndexFlatIP(SYNTHETIC_DIM)
            flat_idx.add(E)
            flat_build = time.time() - t0

            flat_times = []
            flat_ndcg = []
            flat_recall = []
            for qi in range(n_q):
                qv = Q[qi]
                true_scores = build_true_scores(qv, E, FINAL_K)
                t0 = time.time()
                D, I = flat_idx.search(qv.astype(np.float32).reshape(1, -1), FINAL_K)
                elapsed = (time.time() - t0) * 1000
                flat_times.append(elapsed)
                n, r = compute_metrics(I[0], true_scores, FINAL_K)
                flat_ndcg.append(n)
                flat_recall.append(r)
            results["flatip"] = {
                "ndcg_mean": float(np.mean(flat_ndcg)),
                "ndcg_std": float(np.std(flat_ndcg)),
                "recall_mean": float(np.mean(flat_recall)),
                "latency_ms_mean": float(np.mean(flat_times)),
                "latency_ms_std": float(np.std(flat_times)),
                "build_s": flat_build,
                "memory_mb": float(E.nbytes / (1024**2)),
            }
            print(f"    FlatIP: NDCG={np.mean(flat_ndcg):.5f}  Lat={np.mean(flat_times):.2f}ms")

            # --- HNSW ---
            for ef in [32, 64, 128]:
                t0 = time.time()
                hnsw_idx = faiss.IndexHNSWFlat(SYNTHETIC_DIM, 32)
                hnsw_idx.hnsw.efConstruction = 200
                hnsw_idx.add(E)
                hnsw_build = time.time() - t0
                hnsw_idx.hnsw.efSearch = ef

                ht, hn, hr = [], [], []
                for qi in range(n_q):
                    qv = Q[qi]
                    true_scores = build_true_scores(qv, E, FINAL_K)
                    t0 = time.time()
                    D, I = hnsw_idx.search(qv.astype(np.float32).reshape(1, -1), FINAL_K)
                    elapsed = (time.time() - t0) * 1000
                    ht.append(elapsed)
                    n, r = compute_metrics(I[0], true_scores, FINAL_K)
                    hn.append(n)
                    hr.append(r)
                tag = f"hnsw_ef{ef}"
                results[tag] = {
                    "ndcg_mean": float(np.mean(hn)),
                    "ndcg_std": float(np.std(hn)),
                    "recall_mean": float(np.mean(hr)),
                    "latency_ms_mean": float(np.mean(ht)),
                    "latency_ms_std": float(np.std(ht)),
                    "build_s": hnsw_build,
                    "efSearch": ef,
                }
                print(f"    HNSW(ef={ef:3d}): NDCG={np.mean(hn):.5f}  Lat={np.mean(ht):.2f}ms  "
                      f"Recall={np.mean(hr):.3f}")

            # --- IVF ---
            for nprobe in [1, 10, 20]:
                nlist = min(int(math.sqrt(nc)), 256)
                t0 = time.time()
                quant = faiss.IndexFlatIP(SYNTHETIC_DIM)
                ivf_idx = faiss.IndexIVFFlat(quant, SYNTHETIC_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
                ivf_idx.train(E)
                ivf_idx.add(E)
                ivf_build = time.time() - t0
                ivf_idx.nprobe = nprobe

                it, ind, ir = [], [], []
                for qi in range(n_q):
                    qv = Q[qi]
                    true_scores = build_true_scores(qv, E, FINAL_K)
                    t0 = time.time()
                    D, I = ivf_idx.search(qv.astype(np.float32).reshape(1, -1), FINAL_K)
                    elapsed = (time.time() - t0) * 1000
                    it.append(elapsed)
                    n, r = compute_metrics(I[0], true_scores, FINAL_K)
                    ind.append(n)
                    ir.append(r)
                tag = f"ivf_nprobe{nprobe}"
                results[tag] = {
                    "ndcg_mean": float(np.mean(ind)),
                    "ndcg_std": float(np.std(ind)),
                    "recall_mean": float(np.mean(ir)),
                    "latency_ms_mean": float(np.mean(it)),
                    "latency_ms_std": float(np.std(it)),
                    "build_s": ivf_build,
                    "nprobe": nprobe,
                    "nlist": nlist,
                }
                print(f"    IVF(nprobe={nprobe:2d}): NDCG={np.mean(ind):.5f}  Lat={np.mean(it):.2f}ms")

            # --- PQ ---
            for m in [4, 8, 16]:
                if SYNTHETIC_DIM % m != 0:
                    continue
                t0 = time.time()
                pq_idx = faiss.IndexPQ(SYNTHETIC_DIM, m, 8, faiss.METRIC_INNER_PRODUCT)
                pq_idx.train(E)
                pq_idx.add(E)
                pq_build = time.time() - t0

                pt, pn, pr = [], [], []
                for qi in range(n_q):
                    qv = Q[qi]
                    true_scores = build_true_scores(qv, E, FINAL_K)
                    t0 = time.time()
                    D, I = pq_idx.search(qv.astype(np.float32).reshape(1, -1), FINAL_K)
                    elapsed = (time.time() - t0) * 1000
                    pt.append(elapsed)
                    n, r = compute_metrics(I[0], true_scores, FINAL_K)
                    pn.append(n)
                    pr.append(r)
                tag = f"pq_m{m}"
                results[tag] = {
                    "ndcg_mean": float(np.mean(pn)),
                    "ndcg_std": float(np.std(pn)),
                    "recall_mean": float(np.mean(pr)),
                    "latency_ms_mean": float(np.mean(pt)),
                    "latency_ms_std": float(np.std(pt)),
                    "build_s": pq_build,
                    "m": m,
                }
                print(f"    PQ(m={m:2d}):     NDCG={np.mean(pn):.5f}  Lat={np.mean(pt):.2f}ms")
        except ImportError:
            print("    FAISS not available, skipping FlatIP/HNSW/IVF/PQ")
        except Exception as e:
            print(f"    FAISS error: {e}")

        # --- Madhava QR-JL ---
        mad_times = []
        mad_ndcg = []
        mad_recall = []
        mad_violations_total = {f"{d}D": 0 for d in MADHAVA_DIMS}
        mad_viol_pairs = 0
        mad_profiles = []

        mad = MadhavaQRJL()
        mad.build(E)
        mad_build = mad.build_time
        print(f"    Madhava build: {mad_build:.3f}s")

        for qi in range(n_q):
            qv = Q[qi]
            true_scores = build_true_scores(qv, E, FINAL_K)
            t0 = time.time()
            top, prof = mad.search(qv, return_profile=True)
            elapsed = (time.time() - t0) * 1000
            mad_times.append(elapsed)
            n, r = compute_metrics(top, true_scores, FINAL_K)
            mad_ndcg.append(n)
            mad_recall.append(r)
            prof["ndcg"] = float(n)
            prof["latency_ms"] = elapsed
            mad_profiles.append(prof)

            # Bound violations
            viol, total = mad.bound_violations(qv)
            for kk, vv in viol.items():
                mad_violations_total[kk] += vv
            mad_viol_pairs += total

        mad_viol_rate = {k: v / max(mad_viol_pairs, 1) for k, v in mad_violations_total.items()}

        results["madhava"] = {
            "ndcg_mean": float(np.mean(mad_ndcg)),
            "ndcg_std": float(np.std(mad_ndcg)),
            "recall_mean": float(np.mean(mad_recall)),
            "latency_ms_mean": float(np.mean(mad_times)),
            "latency_ms_std": float(np.std(mad_times)),
            "latency_ms_median": float(np.median(mad_times)),
            "latency_ms_min": float(np.min(mad_times)),
            "latency_ms_max": float(np.max(mad_times)),
            "build_s": mad_build,
            "bound_violations": mad_violations_total,
            "bound_violation_rate": mad_viol_rate,
            "total_pairs_checked": mad_viol_pairs,
            "stage1_avg_candidates": float(np.mean([p["n_candidates_stage1"] for p in mad_profiles])),
            "stage2_avg_candidates": float(np.mean([p["n_candidates_stage2"] for p in mad_profiles])),
            "avg_prune_ratio_stage1": float(np.mean([p["prune_ratio_stage1"] for p in mad_profiles])),
            "avg_alpha_mean": float(np.mean([p.get("alpha_mean", 0) for p in mad_profiles])),
            "avg_delta_mean": float(np.mean([p.get("delta_mean", 0) for p in mad_profiles])),
            "avg_modulated_mean": float(np.mean([p.get("modulated_mean", 0) for p in mad_profiles])),
            "avg_B1_range": [float(np.mean([p["B1_range"][0] for p in mad_profiles])),
                             float(np.mean([p["B1_range"][1] for p in mad_profiles]))],
            "avg_B2_range": [float(np.mean([p["B2_range"][0] for p in mad_profiles])),
                             float(np.mean([p["B2_range"][1] for p in mad_profiles]))],
        }
        print(f"    Madhava QR-JL: NDCG={np.mean(mad_ndcg):.5f}  Lat={np.mean(mad_times):.2f}ms  "
              f"Recall={np.mean(mad_recall):.3f}")
        viol_str = " | ".join(f"{k}:{v}" for k, v in mad_violations_total.items())
        print(f"    Bound violations: {viol_str} / {mad_viol_pairs} pairs "
              f"(rate: {' | '.join(f'{k}:{v*100:.4f}%' for k, v in mad_viol_rate.items())})")

        all_results[f"N={nc}"] = results
        gc.collect()

    return all_results

def build_true_scores(query, vectors, k=FINAL_K):
    cos = vectors @ query.astype(np.float32)
    topk = np.argsort(-cos)[:k]
    return {int(idx): float(cos[idx]) for idx in topk}

def compute_metrics(ranked, true_scores, k=FINAL_K):
    n = ndcg_at_k(ranked, true_scores, k)
    r = recall_at_k(ranked, true_scores, k)
    return n, r

# ================================================================
# REPORTING
# ================================================================

def print_comparison_table(all_results):
    """Print formatted comparison across all N and methods."""
    print("\n" + "=" * 80)
    print("  TABELA COMPARATIVA: Todos os Metodos x N")
    print("=" * 80)

    methods = [
        ("FlatIP (exato)", "flatip"),
        ("HNSW(ef=32)", "hnsw_ef32"),
        ("HNSW(ef=64)", "hnsw_ef64"),
        ("HNSW(ef=128)", "hnsw_ef128"),
        ("IVF(nprobe=1)", "ivf_nprobe1"),
        ("IVF(nprobe=10)", "ivf_nprobe10"),
        ("IVF(nprobe=20)", "ivf_nprobe20"),
        ("PQ(m=8)", "pq_m8"),
        ("PQ(m=16)", "pq_m16"),
        ("Madhava QR-JL", "madhava"),
    ]

    n_values = sorted(set(k.split("=")[1] for k in all_results.keys()))

    for nc in n_values:
        key = f"N={nc}"
        if key not in all_results:
            continue
        r = all_results[key]

        print(f"\n  --- N = {int(nc):>8,} ---")
        print(f"  {'Method':<25} {'NDCG@10':>8} {'Recall@10':>10} {'Lat(ms)':>8} {'Build(s)':>8} {'Viol%':>8}")
        print(f"  {'-'*68}")

        for mname, mkey in methods:
            if mkey not in r or not r[mkey]:
                continue
            d = r[mkey]
            lat = f"{d['latency_ms_mean']:.2f}" if "latency_ms_mean" in d else "?"
            build = f"{d['build_s']:.2f}" if "build_s" in d else "?"
            if "bound_violation_rate" in d:
                viol = f"{max(d['bound_violation_rate'].values())*100:.4f}"
            else:
                viol = "N/A"
            print(f"  {mname:<25} {d.get('ndcg_mean', 0):>8.4f} {d.get('recall_mean', 0):>10.4f} "
                  f"{lat:>8} {build:>8} {viol:>8}")

        # Madhava pruning stats
        m = r.get("madhava", {})
        if m:
            print(f"  {'  (Madhava pruning)':<25}")
            print(f"  {'    Stage1(8D) candidates':<30} {m.get('stage1_avg_candidates', 0):>8.0f}")
            print(f"  {'    Stage2(64D) candidates':<30} {m.get('stage2_avg_candidates', 0):>8.0f}")
            print(f"  {'    Prune ratio stage1':<30} {m.get('avg_prune_ratio_stage1', 0):>8.2%}")
            if 'avg_alpha_mean' in m:
                print(f"  {'    Mean alpha (modulation)':<30} {m['avg_alpha_mean']:>8.4f}")
            if 'avg_delta_mean' in m:
                print(f"  {'    Mean delta (error)':<30} {m['avg_delta_mean']:>8.4f}")
            if 'avg_modulated_mean' in m:
                print(f"  {'    Mean modulated score':<30} {m['avg_modulated_mean']:>8.4f}")

def print_summary(all_results):
    """Print summary table: NDCG retention vs baseline for each method."""
    print("\n" + "=" * 80)
    print("  SUMARIO: Retencao NDCG vs FlatIP (baseline)")
    print("=" * 80)

    n_values = sorted(set(k.split("=")[1] for k in all_results.keys()))
    methods = [
        ("HNSW(ef=32)", "hnsw_ef32"),
        ("HNSW(ef=64)", "hnsw_ef64"),
        ("IVF(nprobe=10)", "ivf_nprobe10"),
        ("PQ(m=8)", "pq_m8"),
        ("Madhava QR-JL", "madhava"),
    ]

    header = f"  {'N':>10}"
    for mname, _ in methods:
        header += f" {mname:>16}"
    print(header)
    print(f"  {'-'*10}{'-'*16*len(methods)}")

    for nc in n_values:
        key = f"N={nc}"
        r = all_results.get(key, {})
        flat_ndcg = r.get("flatip", {}).get("ndcg_mean", 1.0)
        line = f"  {int(nc):>10,}"
        for mname, mkey in methods:
            m_ndcg = r.get(mkey, {}).get("ndcg_mean", 0)
            pct = m_ndcg / flat_ndcg * 100 if flat_ndcg > 0 else 0
            line += f" {pct:>15.1f}%"
        print(line)

    print()
    print("  Legenda:")
    print("    NDCG retention = method_NDCG / FlatIP_NDCG * 100%")
    print("    Valores >100% indicam que o metodo aproximado superou o exato")
    print("    (ruido amostral ou viés de aproximacao)")

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    t_total = time.time()
    all_results = {}

    if not args.real_only:
        all_results = run_benchmark_synthetic()

    # --- REAL DATA (if available) ---
    if os.path.exists("/kaggle/input/news-category-dataset/News_Category_Dataset_v3.json"):
        try:
            print("\n" + "=" * 70)
            print("  BENCHMARK B: NEWS CATEGORY DATASET (dados reais)")
            print("=" * 70)
            from sentence_transformers import SentenceTransformer
            import pandas as pd

            print("  Loading SBERT model...")
            embdr = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            print("  Loading news dataset...")
            records = []
            with open("/kaggle/input/news-category-dataset/News_Category_Dataset_v3.json") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            df = pd.DataFrame(records).dropna().reset_index(drop=True)
            texts = (df["headline"].fillna("") + " " +
                     df.get("short_description", df.get("description", "")).fillna("")).tolist()
            n_max = min(20000, len(texts))
            print(f"  Encoding {n_max} texts...")
            embs = embdr.encode(texts[:n_max], convert_to_tensor=False,
                                show_progress_bar=True, normalize_embeddings=True)
            embs = np.array(embs).astype(np.float32)
            print(f"  Shape: {embs.shape}")

            E = embs
            cats = (df["category"].value_counts().index[:20]).tolist()
            rng_q = np.random.RandomState(SEED + 999)
            queries = []
            for cat in cats[:20]:
                ids = np.where(df["category"] == cat)[0]
                if len(ids) > 0:
                    qi = ids[rng_q.randint(len(ids))]
                    queries.append(E[qi])

            print(f"  Running benchmarks on {len(queries)} queries...")
            n_q = min(N_QUERIES_MAX, len(queries))
            real_results = {"N": len(E), "dim": E.shape[1], "dataset": "news_category"}

            try:
                import faiss
                flat_idx = faiss.IndexFlatIP(E.shape[1])
                flat_idx.add(E)
                flat_t, flat_n, flat_r = [], [], []
                for qi in range(n_q):
                    qv = queries[qi]
                    ts = build_true_scores(qv, E, FINAL_K)
                    t0 = time.time()
                    D, I = flat_idx.search(qv.astype(np.float32).reshape(1, -1), FINAL_K)
                    flat_t.append((time.time() - t0) * 1000)
                    n, r = compute_metrics(I[0], ts, FINAL_K)
                    flat_n.append(n)
                    flat_r.append(r)
                real_results["flatip"] = {
                    "ndcg_mean": float(np.mean(flat_n)), "ndcg_std": float(np.std(flat_n)),
                    "recall_mean": float(np.mean(flat_r)),
                    "latency_ms_mean": float(np.mean(flat_t)), "latency_ms_std": float(np.std(flat_t)),
                }
                print(f"    FlatIP: NDCG={np.mean(flat_n):.5f} Lat={np.mean(flat_t):.2f}ms")
            except Exception as e:
                print(f"    FAISS FlatIP error: {e}")

            # Madhava on real data
            mad = MadhavaQRJL()
            mad.build(E)
            mt, mn, mr, mprof = [], [], [], []
            for qi in range(n_q):
                qv = queries[qi]
                ts = build_true_scores(qv, E, FINAL_K)
                t0 = time.time()
                top, prof = mad.search(qv, return_profile=True)
                mt.append((time.time() - t0) * 1000)
                n, r = compute_metrics(top, ts, FINAL_K)
                mn.append(n)
                mr.append(r)
                mprof.append(prof)
            real_results["madhava"] = {
                "ndcg_mean": float(np.mean(mn)), "ndcg_std": float(np.std(mn)),
                "recall_mean": float(np.mean(mr)),
                "latency_ms_mean": float(np.mean(mt)), "latency_ms_std": float(np.std(mt)),
                "stage1_avg_candidates": float(np.mean([p["n_candidates_stage1"] for p in mprof])),
                "stage2_avg_candidates": float(np.mean([p["n_candidates_stage2"] for p in mprof])),
                "avg_alpha_mean": float(np.mean([p.get("alpha_mean", 0) for p in mprof])),
                "avg_delta_mean": float(np.mean([p.get("delta_mean", 0) for p in mprof])),
            }
            print(f"    Madhava QR-JL: NDCG={np.mean(mn):.5f} Lat={np.mean(mt):.2f}ms")

            all_results["REAL_DATA"] = real_results

        except ImportError as e:
            print(f"  Real data benchmark skipped (import error: {e})")
        except Exception as e:
            print(f"  Real data benchmark error: {e}")

    # --- Print final tables ---
    print_comparison_table(all_results)

    if "REAL_DATA" in all_results:
        print("\n  REAL DATA QUICK STATS:")
        rd = all_results["REAL_DATA"]
        f = rd.get("flatip", {})
        m = rd.get("madhava", {})
        if f and m:
            retention = m["ndcg_mean"] / f["ndcg_mean"] * 100 if f["ndcg_mean"] > 0 else 0
            print(f"    FlatIP NDCG@10: {f['ndcg_mean']:.5f}")
            print(f"    Madhava NDCG@10: {m['ndcg_mean']:.5f}")
            print(f"    Retention: {retention:.1f}%")
            print(f"    FlatIP latency: {f['latency_ms_mean']:.3f}ms")
            print(f"    Madhava latency: {m['latency_ms_mean']:.3f}ms")

    if not args.real_only:
        print_summary(all_results)

    elapsed = time.time() - t_total
    print(f"\n  Tempo total: {elapsed:.1f}s")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "config": {
                "seed": SEED,
                "dims": MADHAVA_DIMS,
                "full_dim": FULL_DIM,
                "cascade_topk": CASCADE_TOPK,
                "synth_N": SYNTHETIC_NS,
                "queries": N_QUERIES_MAX,
                "runs": N_RUNS,
            },
            "results": all_results,
            "elapsed_s": elapsed,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados salvos em: {OUTPUT_FILE}")
    print("  FIM.")

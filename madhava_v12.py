#!/usr/bin/env python3
"""
Madhava Adaptive v12: Configurable Enterprise Vector Search

Reads config.json for all parameters. Supports dynamic stage dimensions,
adaptive keep-ratio, hybrid clustering, and error backprop modulation.

Usage:
    python madhava_v12.py                    # benchmark with defaults
    python madhava_v12.py --config config.json
    python madhava_v12.py --quick            # smaller test
    MODULE: from madhava_v12 import MadhavaFactory, MadhybridFactory

License: BSL 1.1 | pay@winnex.ai
"""
import json, os, math, time, random, gc, sys
import numpy as np

# ================================================================
# CONFIG LOADER
# ================================================================
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config(path=None):
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return _default_config()
    with open(path) as f:
        cfg = json.load(f)
    # Merge with defaults for missing keys
    defaults = _default_config()
    cfg = _deep_merge(defaults, cfg)
    return cfg

def _default_config():
    return {
        "version": "12.0.0",
        "dimensions": {"input_dim": 128, "stage_dims": [64, 128], "qjl_dim": 128},
        "search": {
            "adaptive_keep_base": 0.25, "adaptive_keep_min": 0.05,
            "adaptive_keep_max": 0.50, "adaptive_bounds_sensitivity": 0.12,
            "stage2_topk": 500, "stage2_topk_max": 2000, "final_results": 10,
            "epsilon": 1e-5
        },
        "hybrid": {"enabled": True, "n_cells": 64, "n_probe": [3,5,8,10,15],
                   "clustering": {"algorithm": "MiniBatchKMeans", "random_state": 42,
                                  "batch_size": 20000, "n_init": 3, "max_iter": 50}},
        "bounds": {"cauchy_schwarz_epsilon": 1e-5, "orthogonality_tolerance": 1e-5},
        "modulation": {"error_backprop": True, "alpha_smoothing": 0.5,
                       "alpha_min": 0.01, "alpha_max": 0.99},
        "benchmark": {"dataset": "sift-1m", "n_queries": 500, "recall_k": 10},
        "streaming": {"max_rebuild_frequency_ms": 5000, "dynamic_insert_batch": 1000,
                      "cache_projections": True},
        "logging": {"level": "INFO", "results_file": "benchmark_results.json", "timing": True}
    }

def _deep_merge(base, override):
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

# ================================================================
# MADHAVA CORE (config-driven dimensions)
# ================================================================
class MadhavaCore:
    """
    Core Madhava search unit. Configurable dimensions:
      stage_dims=[32,64] -> original 32D->64D->exact
      stage_dims=[64,128] -> high-res 64D->128D->exact
    All parameters from config.json.
    """
    def __init__(self, config=None):
        self.cfg = config or load_config()
        self.dims = self.cfg['dimensions']['stage_dims']
        self.full_dim = self.cfg['dimensions']['input_dim']
        self.s = self.cfg['search']
        self.b = self.cfg['bounds']
        self.m = self.cfg['modulation']
        self.rng = np.random.RandomState(43)
        self.vectors = None; self.proj_L = {}; self.error = {}
        self.proj_matrices = {}; self.build_time = 0.0

    def _make_orthogonal_proj(self, d_out):
        Q,_ = np.linalg.qr(self.rng.randn(d_out, self.full_dim).astype(np.float64).T)
        P = Q[:,:d_out].T.astype(np.float64)
        err = np.abs(P @ P.T - np.eye(d_out)).max()
        assert err < self.b['orthogonality_tolerance'], f"QR failed: {err:.2e}"
        return P

    def build(self, vectors):
        t0 = time.time()
        self.vectors = vectors.astype(np.float64)
        self.n = len(vectors)
        self.norms = np.linalg.norm(self.vectors, axis=1)
        for d in self.dims:
            P = self._make_orthogonal_proj(d)
            self.proj_matrices[d] = P
            proj = (vectors.astype(np.float32) @ P.T.astype(np.float32)).astype(np.float64)
            self.proj_L[d] = proj
            captured = np.linalg.norm(proj, axis=1)
            self.error[d] = np.sqrt(np.maximum(self.norms**2 - captured**2, 0))
        self.build_time = time.time() - t0
        return self

    def _upper_bound(self, pv, ev, pq, eq):
        eps = self.b['cauchy_schwarz_epsilon']
        return pv @ pq + ev * eq + eps

    def search(self, q, k=None):
        if k is None: k = self.s['final_results']
        if self.n == 0: return np.array([], dtype=int)
        q = q.astype(np.float64).flatten()
        qn = np.linalg.norm(q)
        p = {}

        # Stage 1: lowest dim on ALL N
        d1 = self.dims[0]
        q1 = (q.astype(np.float32) @ self.proj_matrices[d1].T.astype(np.float32)).astype(np.float64)
        qr1 = math.sqrt(max(0, qn**2 - np.linalg.norm(q1)**2))
        B1 = self._upper_bound(self.proj_L[d1], self.error[d1], q1, qr1)

        # Adaptive keep based on bound range
        b_range = float(B1.max() - B1.min())
        base = self.s['adaptive_keep_base']
        sens = self.s['adaptive_bounds_sensitivity']
        raw_keep = base * sens / max(b_range, 0.01)
        adapt_k = min(self.s['adaptive_keep_max'],
                      max(self.s['adaptive_keep_min'], raw_keep))
        k1 = min(max(int(self.n * adapt_k), 100), self.n)
        if self.n <= k1:
            idx1 = np.arange(self.n)
        else:
            idx1 = np.argpartition(-B1, k1 - 1)[:k1]

        # Stage 2: higher dim refinement
        d2 = self.dims[1]
        q2 = (q.astype(np.float32) @ self.proj_matrices[d2].T.astype(np.float32)).astype(np.float64)
        qr2 = math.sqrt(max(0, qn**2 - np.linalg.norm(q2)**2))
        B2 = self._upper_bound(self.proj_L[d2][idx1], self.error[d2][idx1], q2, qr2)

        # Error backprop modulation
        if self.m['error_backprop']:
            e1 = self.error[d1][idx1]
            e2 = self.error[d2][idx1]
            alpha = 1.0 / (1.0 + np.exp(-(e1 - e2) / max(np.mean(e1), 1e-9) * self.m['alpha_smoothing']))
            alpha = np.clip(alpha, self.m['alpha_min'], self.m['alpha_max'])
            scores = B1[idx1] + alpha * (B2 - B1[idx1])
        else:
            scores = B2

        # Stage 3: exact cosine on top-k2 survivors
        k2 = min(self.s['stage2_topk'], len(idx1))
        idx2 = idx1[np.argpartition(-scores, k2 - 1)[:k2]]
        cos = self.vectors[idx2].astype(np.float64) @ q
        return idx2[np.argsort(-cos)[:k]]

# ================================================================
# MADHYBRID (clustered + Madhava per cell)
# ================================================================
class Madhybrid:
    def __init__(self, config=None):
        self.cfg = config or load_config()
        self.nc = self.cfg['hybrid']['n_cells']
        self.clust_cfg = self.cfg['hybrid']['clustering']

    def build(self, vectors):
        from sklearn.cluster import MiniBatchKMeans
        t0 = time.time()
        self.vecs = vectors
        bs = min(self.clust_cfg['batch_size'], len(vectors))
        km = MiniBatchKMeans(n_clusters=self.nc, random_state=self.clust_cfg['random_state'],
                             batch_size=bs, n_init=self.clust_cfg['n_init'],
                             max_iter=self.clust_cfg['max_iter'])
        labs = km.fit_predict(vectors)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self.cells = {}; self.members = {}
        for cid in range(self.nc):
            idxs = np.where(labs == cid)[0]
            if len(idxs) == 0: continue
            self.members[cid] = idxs
            c = MadhavaCore(self.cfg); c.build(vectors[idxs])
            self.cells[cid] = c
        self.build_time = time.time() - t0
        return self

    def search(self, q, k=None, np_=None):
        if np_ is None: np_ = self.cfg['hybrid']['n_probe'][0]
        if k is None: k = self.cfg['search']['final_results']
        q = q.astype(np.float32).flatten()
        sims = self.centroids @ q
        top_c = np.argsort(-sims)[:np_]
        candidates = []
        for cid in top_c:
            c = self.cells.get(cid)
            if c is None or c.n == 0: continue
            idxs = c.search(q, k)
            scores = c.vectors[idxs].astype(np.float64) @ q.astype(np.float64)
            for i_, s_ in zip(idxs, scores):
                candidates.append((self.members[cid][i_], float(s_)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:k]]

# ================================================================
# BENCHMARK
# ================================================================
def load_sift(n=100000, nq=500):
    try:
        import h5py
        if not os.path.exists('/tmp/sift.hdf5'):
            import urllib.request
            print("Downloading SIFT-1M (~525MB)...")
            urllib.request.urlretrieve(
                'http://ann-benchmarks.com/sift-128-euclidean.hdf5', '/tmp/sift.hdf5')
        with h5py.File('/tmp/sift.hdf5', 'r') as f:
            E = f['train'][:n].astype(np.float32)
            Q = f['test'][:nq].astype(np.float32)
    except:
        print("SIFT unavailable, generating structured 100K...")
        N=100000; NC=256; D=128
        centers = np.random.randn(NC, D).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        X = []
        for ci in range(NC):
            cnt = N//NC + (1 if ci < N%NC else 0)
            pts = centers[ci] + np.random.randn(cnt, D).astype(np.float32)*0.25
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            X.append(pts)
        E = np.vstack(X).astype(np.float32)
        qi = np.random.RandomState(42).choice(N, nq, replace=False)
        Q = E[qi].copy()
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    return E, Q

def run_benchmark(config=None, quick=False):
    cfg = config or load_config()
    print(f"\nMadhava v12 — Config: stage_dims={cfg['dimensions']['stage_dims']}")
    print(f"{'='*70}")

    N = 50000 if quick else 100000
    NQ = 200 if quick else cfg['benchmark']['n_queries']
    K = cfg['benchmark']['recall_k']

    print(f"Loading data ({N} vectors, {NQ} queries)...")
    E, Q = load_sift(N, NQ)
    D = E.shape[1]

    # Ground truth
    import faiss
    fi = faiss.IndexFlatIP(D); fi.add(E)
    GT = np.zeros((NQ, K), dtype=np.int32)
    for qi in range(NQ):
        _, I = fi.search(Q[qi:qi+1], K)
        GT[qi] = I[0]

    def recall_k(retrieved, qi):
        return len(set(retrieved[:K]) & set(GT[qi][:K])) / K

    print(f"{'Method':>30} {'R@10':>8} {'Lat(ms)':>10} {'QPS':>10} {'Build(s)':>10}")
    print(f"{'─'*68}")

    # HNSW
    idx = faiss.IndexHNSWFlat(D, 32); idx.hnsw.efConstruction = 200
    t0 = time.time(); idx.add(E); hb = time.time() - t0
    for ef in [64, 128, 256]:
        idx.hnsw.efSearch = ef
        ht, hr = [], []
        for qi in range(NQ):
            t0 = time.time(); _, I = idx.search(Q[qi:qi+1], K)
            ht.append((time.time()-t0)*1000); hr.append(recall_k(I[0], qi))
        print(f"{'HNSW(ef='+str(ef)+')':>30} {np.mean(hr):>8.4f} {np.mean(ht):>10.3f} {1000/np.mean(ht):>9.0f} {hb:>8.1f}s")

    # IVF
    for npb in [5, 10, 20]:
        qf = faiss.IndexFlatIP(D)
        ivf = faiss.IndexIVFFlat(qf, D, 256, faiss.METRIC_INNER_PRODUCT)
        ivf.train(E); ivf.add(E); ivf.nprobe = npb
        it, ir = [], []
        for qi in range(NQ):
            t0 = time.time(); _, I = ivf.search(Q[qi:qi+1], K)
            it.append((time.time()-t0)*1000); ir.append(recall_k(I[0], qi))
        print(f"{'IVF(nprobe='+str(npb)+')':>30} {np.mean(ir):>8.4f} {np.mean(it):>10.3f} {1000/np.mean(it):>9.0f} {'<1m':>10}")

    # Standalone Madhava
    mc = MadhavaCore(cfg)
    mc.build(E)
    mt, mr = [], []
    for qi in range(NQ):
        t0 = time.time(); top = mc.search(Q[qi], K)
        mt.append((time.time()-t0)*1000); mr.append(recall_k(top, qi))
    print(f"{'MadhavaCore(1-cell)':>30} {np.mean(mr):>8.4f} {np.mean(mt):>10.3f} {1000/np.mean(mt):>9.0f} {mc.build_time:>8.3f}s")

    # MadHybrid
    if cfg['hybrid']['enabled']:
        mh = Madhybrid(cfg); mh.build(E)
        for np_ in cfg['hybrid']['n_probe']:
            mt, mr = [], []
            for qi in range(NQ):
                t0 = time.time(); top = mh.search(Q[qi], K, np_)
                mt.append((time.time()-t0)*1000); mr.append(recall_k(top, qi))
            print(f"{'MadHybrid(np='+str(np_)+')':>30} {np.mean(mr):>8.4f} {np.mean(mt):>10.3f} {1000/np.mean(mt):>9.0f} {mh.build_time:>8.3f}s")

    print(f"\n{'='*70}")
    print(f"Streaming capability: {int(60/mh.build_time)} rebuilds/minute")
    print(f"Build ratio: HNSW({hb:.1f}s) / MadHybrid({mh.build_time:.1f}s) = {hb/mh.build_time:.1f}x")
    gc.collect()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=None)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_benchmark(cfg, quick=args.quick)

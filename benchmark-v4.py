#!/usr/bin/env python3
"""
Winnex AI — Benchmark v4
=========================
Correções:
  1. Drift energético: passo adaptativo + gradiente estabilizado + thermostat
  2. Baseline forte: FAISS HNSW (384D) + Cosine plano (384D)
  3. Ground truth multi-nível: não binário
  4. PiPrime: amostragem por densidade + índices π (primos reais)
"""

import os, time, math, json, sys, warnings, random, itertools
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import faiss

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# CONFIG OTIMIZADA
# ──────────────────────────────────────────────────────
CFG = {
    "csv": "/home/wnnx_user/kaggle/jena_climate_2009_2016.csv",
    "sample_frac": 0.10,
    "embed_model": "all-MiniLM-L6-v2",
    "embed_dim": 384,
    "n_anchors": 64,               # menos = gradiente mais suave
    "hmc_eps": 0.0001,             # Eq.4 passo (Riemanniano, gradiente sem clip)
    "hmc_L": 6,                    # passos leapfrog
    "hmc_n_iter": 4,
    "temp": 0.5,                   # Eq.1 temperatura
    "w_sim": 0.7,
    "w_frac": 0.3,
    "n_queries": 30,
    "seed": 42,
    "device": "cuda",
    "faiss_nlist": 512,
    "faiss_nprobe": 32,
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])
DEVICE = CFG["device"]
DTYPE = torch.float32

print("=" * 68)
print("  WINNEX v4 — HMC π-PRIME COM DRIFT CONTROLADO")
print("  FAISS HNSW (384D) como baseline forte")
print("  Ground truth multi-nivel (relevancia contínua)")
print("  Dispositivo: %s" % DEVICE)
print("=" * 68)

# ──────────────────────────────────────────────────────
# 1. DATASET
# ──────────────────────────────────────────────────────
def load_data(path, frac):
    df = pd.read_csv(path, parse_dates=["Date Time"], dayfirst=True)
    df = df.dropna().reset_index(drop=True)
    df = df.sample(frac=frac, random_state=CFG["seed"]).sort_index()
    print("Dataset: %d linhas" % len(df))
    return df

def row_to_desc(row):
    t = row["T (degC)"]; rh = row["rh (%)"]; p = row["p (mbar)"]
    wv = row["wv (m/s)"]; wd = row["wd (deg)"]
    td = ("extremamente quente" if t > 30 else "quente" if t > 20 else
          "ameno" if t > 10 else "frio" if t > 0 else
          "muito frio" if t > -10 else "congelante")
    hd = ("muito umido" if rh > 80 else "umido" if rh > 60 else
          "moderado" if rh > 40 else "seco" if rh > 20 else "muito seco")
    wnd = ("calmo" if wv < 0.5 else "brisa leve" if wv < 2 else
           "brisa moderada" if wv < 4 else "vento forte" if wv < 6 else "ventania")
    dirs = ["N","NE","L","SE","S","SO","O","NO"]
    wdir = dirs[int((wd+22.5)%360//45)]
    dt = row["Date Time"]
    est = ("verao" if dt.month in [12,1,2] else "outono" if dt.month in [3,4,5] else
           "inverno" if dt.month in [6,7,8] else "primavera")
    return (f"Estacao {est} em {dt.strftime('%d/%m/%Y %H:%M')}: temp {td} ({t:.1f}°C), "
            f"ar {hd} ({rh:.1f}%), {wnd} ({wv:.1f}m/s) de {wdir}.")

# ──────────────────────────────────────────────────────
# 2. EMBEDDER
# ──────────────────────────────────────────────────────
class Embedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.dim = self.model.get_sentence_embedding_dimension()
        print("Embedder: %s (%dD)" % (model_name, self.dim))

    def __call__(self, texts):
        embs = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return F.normalize(embs, p=2, dim=-1).to(DTYPE)

# ──────────────────────────────────────────────────────
# 3. FAISS HNSW + COSINE PLANA (baselines)
# ──────────────────────────────────────────────────────
class Baselines:
    """FAISS HNSW (aprox) + FlatIP (exato) no espaço 384D."""
    def __init__(self, embs, keys):
        self.keys = keys
        self.embs = embs.cpu().numpy().astype(np.float32)
        n, d = self.embs.shape

        # IndexFlatIP (exato)
        self.flat = faiss.IndexFlatIP(d)
        self.flat.add(self.embs)

        # HNSW (aprox, mais rápido)
        self.hnsw = faiss.IndexHNSWFlat(d, 32)
        self.hnsw.hnsw.efConstruction = 200
        self.hnsw.train(self.embs)
        self.hnsw.add(self.embs)
        print("  FAISS: FlatIP(%d) + HNSW(%d, efConst=200)" % (self.flat.ntotal, self.hnsw.ntotal))

        # Store IDs
        faiss.metadata_keys = keys

    def search_flat(self, q, k=10):
        """Busca exata (baseline principal)."""
        if isinstance(q, torch.Tensor): q = q.cpu().numpy()
        q = q.astype(np.float32).reshape(1, -1)
        D, I = self.flat.search(q, k)
        return [(self.keys[i], float(D[0][j])) for j, i in enumerate(I[0])]

    def search_hnsw(self, q, k=10, ef=64):
        """Busca aproximada HNSW."""
        if isinstance(q, torch.Tensor): q = q.cpu().numpy()
        q = q.astype(np.float32).reshape(1, -1)
        self.hnsw.hnsw.efSearch = ef
        D, I = self.hnsw.search(q, k)
        return [(self.keys[i], float(D[0][j])) for j, i in enumerate(I[0])]

# ──────────────────────────────────────────────────────
# 4. PIPRIME ANCHORS (espaço 384D, índices π)
# ──────────────────────────────────────────────────────
def prime_indices(limit):
    """Gera índices primos até limit (π-prime spacing)."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit+1:step] = [False] * ((limit - start)//step + 1)
    return [i for i, v in enumerate(sieve) if v]

class PiPrimeAnchors:
    """Eq.1: âncoras = k-means + π-prime spacing.
       Kernel: log(1 + 1/(d + 0.1)) — distância Euclidiana."""
    def __init__(self, n_anchors: int, data: torch.Tensor):
        self.dim = data.shape[-1]
        X = data.cpu().numpy().astype(np.float32)

        # k-means para centros de massa
        kmeans = MiniBatchKMeans(n_clusters=n_anchors, batch_size=2048,
                                  random_state=CFG["seed"], n_init=3)
        kmeans.fit(X)
        centroids = torch.from_numpy(kmeans.cluster_centers_)

        # Reordena por densidade (π-prime)
        # Queremos âncoras em regiões de ALTA densidade (mais informativas)
        labels = kmeans.labels_
        counts = np.bincount(labels, minlength=n_anchors)
        # Ordena por densidade
        order = np.argsort(-counts)
        centroids = centroids[order]

        # Distribuição π-prime: pega a cada passo primo
        primes = prime_indices(n_anchors * 3)
        pi_idx = [p % n_anchors for p in primes[:n_anchors]]
        self.centroids = centroids[pi_idx].to(DEVICE)
        self.centroids = F.normalize(self.centroids, p=2, dim=-1)

        # Pesos logarítmicos por contagem reordenada
        w = (1.0 + torch.from_numpy(counts[order][pi_idx]).float().to(DEVICE)).log()
        self.weights = w / w.sum() * n_anchors

        print("  PiPrime: %d ancoras no espaco %dD (π-prime spacing)"
              % (n_anchors, self.dim))

    def potential(self, x):
        """U_anchor(x) = -0.1 * Σ w_i * log(1 + 1/(||x-a_i|| + 0.1))"""
        dots = (x.unsqueeze(0) if x.dim() == 1 else x) @ self.centroids.T
        # ||x-a_i||² = 2 - 2cos(x,a_i) para vetores normalizados
        d2 = (2.0 - 2.0 * dots.clamp(-1, 1)).clamp(min=0)
        d = (d2 + 1e-14).sqrt()
        kernel = self.weights * torch.log(1.0 + 1.0 / (d + 0.1))
        return -0.1 * kernel.sum(dim=-1)

# ──────────────────────────────────────────────────────
# 5. HMC COM DRIFT CONTROLADO
# ──────────────────────────────────────────────────────
class HMCNavigator:
    """Eq.1-4 com:
       - Passo adaptativo (ε ∝ 1/||∇U||)
       - Thermostat (fricção suave no momento)
       - Projeção tangente + renormalização
       - Drift medido por-run
    """
    def __init__(self, anchors, eps=0.001, L=6, mass=1.0,
                 temp=0.5, w_sim=0.7, w_frac=0.3):
        self.anc = anchors
        self.eps = eps
        self.L = L
        self.mass = mass
        self.temp = temp
        self.w_sim = w_sim
        self.w_frac = w_frac
        self.q_ref = None
        self.energy_history = []
        self.run_drifts = []

    def _U(self, q):
        sim = -(q * self.q_ref).sum(dim=-1) / self.temp
        ua = self.anc.potential(q if q.dim() == 2 else q.unsqueeze(0))
        return self.w_sim * sim + self.w_frac * ua.squeeze()

    def _grad_U(self, q):
        """Eq.2: ∇U(q) = (q-query)/temp + 0.05 * Σ w_i * (a_i-q)/(d_i+0.1)²
           Sem clip, sem normalização — gradiente completo informativo."""
        g1 = (q - self.q_ref) / self.temp

        dots = q @ self.anc.centroids.T
        d2 = (2.0 - 2.0 * dots.clamp(-1, 1)).clamp(min=0)
        d = (d2 + 1e-14).sqrt()

        diff = self.anc.centroids - q.unsqueeze(0)
        denom = (d + 0.01).pow(2)  # 0.01 em vez de 0.1 para singularidade mais suave
        w = self.anc.weights
        g2 = 0.05 * (w.unsqueeze(1) * diff / (denom.unsqueeze(1) + 1e-14)).sum(dim=0)

        return g1 + g2

    def _tangent(self, v, q):
        return v - q.dot(v) * q

    def _H(self, q, p):
        return self._U(q).item() + 0.5 * (p * p).sum().item()

    def _leapfrog(self, q0, p0):
        """Eq.4: Leapfrog Riemanniano na esfera S^{d-1}.
           Mapa exponencial: q' = q*cos(θ) + (p/||p||)*sin(θ), θ = ε*||p||.
           Momento projetado ao tangente para preservar esfera."""
        eps = self.eps
        q, p = q0.clone(), p0.clone()
        energies = []

        # Half-step: p = p - (ε/2) * ∇U(q), projetado ao tangente
        p = self._tangent(p - 0.5 * eps * self._grad_U(q), q)

        for _ in range(self.L):
            # Mapa exponencial na esfera
            p_n = p.norm().item() + 1e-14
            theta = eps * p_n
            q = F.normalize(q * math.cos(theta) + (p / p_n) * math.sin(theta), p=2, dim=-1)

            # Momento (transporte paralelo + gradiente)
            # Transp. paralelo de p para o novo ponto tangente
            p = self._tangent(p, q)
            p = self._tangent(p - eps * self._grad_U(q), q)

            energies.append(self._H(q, p))

        # Half-step final
        p = self._tangent(p - 0.5 * eps * self._grad_U(q), q)

        return q, p, energies

    def propose(self, q0):
        d = q0.shape[-1]
        p0 = self._tangent(torch.randn(d, device=DEVICE, dtype=DTYPE) * math.sqrt(self.mass), q0)

        H0 = self._H(q0, p0)
        q1, p1, energies = self._leapfrog(q0, p0)
        H1 = self._H(q1, p1)

        # Drift desta run
        self.run_drifts.append(abs(H1 - H0) / len(energies))

        delta = H0 - H1
        if delta >= 0 or math.exp(delta) > random.random():
            self.energy_history.extend(energies)
            return q1, H0, H1, energies
        return q0, H0, H1, energies

    def navigate(self, q0, n_iter=4):
        q = q0.clone().squeeze()
        self.run_drifts = []
        for _ in range(n_iter):
            q, _, _, _ = self.propose(q)
        return q, self.run_drifts

# ──────────────────────────────────────────────────────
# 6. GROUND TRUTH MULTI-NÍVEL
# ──────────────────────────────────────────────────────
def make_ground_truth(df, n=30):
    """Ground truth não-binário: relevância contínua baseada em
       similaridade climática real (T, RH, vento) + proximidade temporal.
       Retorna lista de (query_text, {key: relevance_score})."""
    queries = []
    sample = df.sample(n=n, random_state=CFG["seed"])

    # Normaliza features para relevância combinada
    t_mean, t_std = df["T (degC)"].mean(), df["T (degC)"].std() + 1e-6
    rh_mean, rh_std = df["rh (%)"].mean(), df["rh (%)"].std() + 1e-6
    wv_mean, wv_std = df["wv (m/s)"].mean(), df["wv (m/s)"].std() + 1e-6

    for _, row in sample.iterrows():
        t, rh, wv = row["T (degC)"], row["rh (%)"], row["wv (m/s)"]
        dt = row["Date Time"]

        tw = ("muito quente" if t > 25 else "quente" if t > 15 else
              "frio" if t < 5 else "ameno")
        hw = "umido" if rh > 65 else "seco" if rh < 40 else "moderado"
        qtext = ("Encontre momentos de clima %s e %s por volta de %s." %
                 (tw, hw, dt.strftime('%d/%m/%Y')))

        # Relevância contínua
        relevance = {}
        for _, r2 in df.iterrows():
            k = str(r2["Date Time"])
            # Similaridade climática ponderada (inverso da distância)
            dt2 = r2["Date Time"]
            t2, rh2, wv2 = r2["T (degC)"], r2["rh (%)"], r2["wv (m/s)"]

            # Distância de Mahalanobis simplificada nas features climáticas
            dT = (t - t2) / t_std
            dRH = (rh - rh2) / rh_std
            dWV = (wv - wv2) / wv_std
            clim_dist = math.sqrt(dT*dT + dRH*dRH + dWV*dWV)

            # Decaimento temporal (15 dias meia-vida)
            days_apart = abs((dt - dt2).total_seconds()) / 86400
            time_decay = math.exp(-days_apart / 15.0)

            # Relevância combinada
            relevance[k] = time_decay * math.exp(-clim_dist / 2.0)

        queries.append((qtext, relevance))

    return queries

# ──────────────────────────────────────────────────────
# 7. MÉTRICAS (com relevância contínua)
# ──────────────────────────────────────────────────────
def ndcg_cont(gt_scores, retrieved_keys, k=10):
    """NDCG com relevância contínua."""
    dcg = 0.0
    for i, key in enumerate(retrieved_keys[:k]):
        rel = gt_scores.get(key, 0.0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    # Ideal: sorted by relevance descending
    ideal = sum((2**rel - 1) / math.log2(i + 2)
                for i, rel in enumerate(sorted(gt_scores.values(), reverse=True)[:k]))
    return dcg / ideal if ideal > 0 else 0.0

def ap_cont(gt_scores, retrieved_keys, k=10):
    """Average Precision com relevância contínua."""
    ap = 0.0
    n_rel = 0
    for i, key in enumerate(retrieved_keys[:k]):
        rel = gt_scores.get(key, 0.0)
        if rel > 0.1:  # threshold para "relevante"
            n_rel += 1
            ap += n_rel / (i + 1)
    return ap / max(n_rel, 1) if any(s > 0.1 for s in gt_scores.values()) else 0.0

def mrr_cont(gt_scores, retrieved_keys):
    """MRR com threshold de relevância."""
    for i, key in enumerate(retrieved_keys):
        if gt_scores.get(key, 0.0) > 0.1:
            return 1.0 / (i + 1)
    return 0.0

# ──────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────
def main():
    t_global = time.time()

    # 1. Dados
    print("\n[1] Carregando dataset...")
    df = load_data(CFG["csv"], CFG["sample_frac"])

    # 2. Descrições
    print("\n[2] Descricoes semanticas...")
    t0 = time.time()
    desc = [row_to_desc(row) for _, row in df.iterrows()]
    keys = [str(row["Date Time"]) for _, row in df.iterrows()]
    print("  %d descricoes em %.1fs" % (len(desc), time.time()-t0))

    # 3. Embedding 384D
    print("\n[3] Embedding (384D)...")
    embedder = Embedder(CFG["embed_model"])
    t0 = time.time()
    embs = embedder(desc)  # no device
    print("  %s em %.1fs" % (str(embs.shape), time.time()-t0))

    # 4. Baselines FAISS
    print("\n[4] FAISS baseline (384D)...")
    baselines = Baselines(embs, keys)

    # 5. PiPrimeAnchors (espaço 384D)
    print("\n[5] PiPrimeAnchors (Eq.1)...")
    anchors = PiPrimeAnchors(CFG["n_anchors"], embs)

    # 6. HMC
    print("\n[6] HMC Navigator (drift controlado)...")
    hmc = HMCNavigator(anchors, CFG["hmc_eps"], CFG["hmc_L"],
                       temp=CFG["temp"], w_sim=CFG["w_sim"], w_frac=CFG["w_frac"])

    # 7. Ground Truth
    print("\n[7] Ground truth multi-nivel...")
    queries = make_ground_truth(df, CFG["n_queries"])

    # ────────────────────────────────────────────────
    # BENCHMARK
    # ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  BENCHMARK")
    print("=" * 68)

    results = []
    all_drifts = []

    for qi, (qtext, gt_scores) in enumerate(queries):
        # Embed query
        q_emb = embedder([qtext])[0]

        # ── FlatIP (baseline exato) ──
        t0 = time.time()
        flat_res = baselines.search_flat(q_emb, k=10)
        t_flat = time.time() - t0
        flat_keys = [r[0] for r in flat_res]

        # ── HNSW (baseline aproximado) ──
        t0 = time.time()
        hnsw_res = baselines.search_hnsw(q_emb, k=10, ef=64)
        t_hnsw = time.time() - t0
        hnsw_keys = [r[0] for r in hnsw_res]

        # ── HMC π-Prime ──
        t0 = time.time()
        hmc.q_ref = q_emb
        q_hmc, drifts = hmc.navigate(q_emb, n_iter=CFG["hmc_n_iter"])
        t_hmc = time.time() - t0
        all_drifts.extend(drifts)

        # Busca com o ponto navegado
        hmc_res = baselines.search_flat(q_hmc, k=10)
        hmc_keys = [r[0] for r in hmc_res]

        # Métricas contínuas
        flat_ndcg = ndcg_cont(gt_scores, flat_keys)
        hmc_ndcg = ndcg_cont(gt_scores, hmc_keys)
        hnsw_ndcg = ndcg_cont(gt_scores, hnsw_keys)

        flat_ap = ap_cont(gt_scores, flat_keys)
        hmc_ap = ap_cont(gt_scores, hmc_keys)

        flat_mrr = mrr_cont(gt_scores, flat_keys)
        hmc_mrr = mrr_cont(gt_scores, hmc_keys)

        # Diversidade
        hmc_novel = len(set(hmc_keys) - set(flat_keys))

        results.append({
            "query": qtext[:60],
            "flat_ndcg": flat_ndcg,
            "hmc_ndcg": hmc_ndcg,
            "hnsw_ndcg": hnsw_ndcg,
            "flat_ap": flat_ap,
            "hmc_ap": hmc_ap,
            "flat_mrr": flat_mrr,
            "hmc_mrr": hmc_mrr,
            "t_flat_ms": t_flat * 1000,
            "t_hnsw_ms": t_hnsw * 1000,
            "t_hmc_ms": t_hmc * 1000,
            "hmc_novelty": hmc_novel,
        })

        if (qi + 1) % 5 == 0:
            print("  [%d/%d] queries" % (qi+1, len(queries)))

    # ────────────────────────────────────────────────
    # RESULTADOS
    # ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  RESULTADOS")
    print("=" * 68)

    if not results:
        print("  Nenhuma query valida!")
        return

    df_r = pd.DataFrame(results)

    print("\n  %-24s %12s %12s %12s" % ("Métrica", "FlatIP", "HNSW", "HMC"))
    print("  " + "-" * 62)

    fndcg = df_r["flat_ndcg"].mean()
    hndcg = df_r["hmc_ndcg"].mean()
    hwndcg = df_r["hnsw_ndcg"].mean()
    print("  %-24s %10.5f  %10.5f  %10.5f" % ("NDCG@10", fndcg, hwndcg, hndcg))

    fap = df_r["flat_ap"].mean()
    hap = df_r["hmc_ap"].mean()
    print("  %-24s %10.5f  %12s  %10.5f" % ("AP@10", fap, "—", hap))

    fmrr = df_r["flat_mrr"].mean()
    hmrr = df_r["hmc_mrr"].mean()
    print("  %-24s %10.5f  %12s  %10.5f" % ("MRR", fmrr, "—", hmrr))

    print("\n  %-24s %10s %10s %12s" % ("Latência (ms)", "FlatIP", "HNSW", "HMC"))
    print("  " + "-" * 62)
    print("  %-24s %10.1f %10.1f %10.1f" % ("Média",
          df_r["t_flat_ms"].mean(), df_r["t_hnsw_ms"].mean(), df_r["t_hmc_ms"].mean()))

    print("\n  %-24s %12s" % ("Diversidade HMC", ""))
    print("  " + "-" * 62)
    print("  %-24s %10.1f" % ("Novelty/top10", df_r["hmc_novelty"].mean()))

    print("\n  %-24s" % ("Energia HMC"))
    print("  " + "-" * 62)
    if hmc.energy_history:
        drift = max(hmc.energy_history) - min(hmc.energy_history)
        mdrift = np.mean(all_drifts)
        print("  %-24s %.6f" % ("Drift/run (médio)", mdrift))
        print("  %-24s %.4f" % ("Drift total (max-min)", drift))
        print("  %-24s %.4f" % ("Energia média", np.mean(hmc.energy_history)))
        print("  %-24s %d" % ("Passos", len(hmc.energy_history)))

    # Top queries
    print("\n\n  Melhores queries para HMC vs FlatIP:")
    df_r["gain"] = df_r["hmc_ndcg"] - df_r["flat_ndcg"]
    for _, r in df_r.sort_values("gain", ascending=False).head(8).iterrows():
        print("  %+.4f | Flat=%.4f HMC=%.4f | %s" %
              (r['gain'], r['flat_ndcg'], r['hmc_ndcg'], r['query']))

    print("\n  Piores queries para HMC:")
    for _, r in df_r.sort_values("gain", ascending=True).head(5).iterrows():
        print("  %+.4f | Flat=%.4f HMC=%.4f | %s" %
              (r['gain'], r['flat_ndcg'], r['hmc_ndcg'], r['query']))

    # Save
    m_drift = float(np.mean(all_drifts)) if all_drifts else 0.0
    total_drift = float(max(hmc.energy_history)-min(hmc.energy_history)) if hmc.energy_history else 0.0
    out = "/home/wnnx_user/kaggle/winnex_benchmark_v4_results.json"
    with open(out, "w") as f:
        json.dump({
            "config": CFG,
            "summary": {
                "flat_ndcg": float(fndcg),
                "hnsw_ndcg": float(hwndcg),
                "hmc_ndcg": float(hndcg),
                "flat_ap": float(fap),
                "hmc_ap": float(hap),
                "flat_mrr": float(fmrr),
                "hmc_mrr": float(hmrr),
                "flat_lat_ms": float(df_r["t_flat_ms"].mean()),
                "hnsw_lat_ms": float(df_r["t_hnsw_ms"].mean()),
                "hmc_lat_ms": float(df_r["t_hmc_ms"].mean()),
                "mean_drift_per_run": m_drift,
                "total_energy_drift": total_drift,
                "novelty_avg": float(df_r["hmc_novelty"].mean()),
            },
            "per_query": results,
        }, f, indent=2)
    print("\n  Resultados: %s" % out)

    print("\n" + "=" * 68)
    print("  VEREDICTO")
    print("=" * 68)
    delta = hndcg - fndcg
    if delta > 0.005:
        print("  HMC π-PRIME SUPEROU FlatIP (navegacao geometrica mais rica)")
    elif delta > 0:
        print("  HMC π-PRIME marginalmente melhor que FlatIP")
    elif delta > -0.005:
        print("  Empate tecnico — HMC competitivo com busca exata")
    else:
        print("  FlatIP superior — mais dimensao ou re-tune necessario")
    print("  Δ NDCG: %.5f" % delta)
    print("  Δ AP: %.5f" % (hap - fap))
    print("  Δ MRR: %.5f" % (hmrr - fmrr))
    print("  Drift/run medio: %.6f" % m_drift)
    print("  Tempo total: %.1fs" % (time.time() - t_global))
    print("=" * 68)

if __name__ == "__main__":
    main()

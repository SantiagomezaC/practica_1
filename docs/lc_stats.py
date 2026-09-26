"""Pruebas estadísticas para comparar clasificadores evaluados sobre el mismo conjunto de prueba.

* DeLong (1988) en su versión rápida O(n log n) de Sun y Xu (2014), y la versión directa O(n^2)
  usada solo para validar la implementación rápida.
* McNemar (1947) con corrección de continuidad o exacta (binomial) cuando b + c < 25.
* Bootstrap pareado (Efron y Tibshirani, 1993) de ΔAUC, ΔAUC-PR y ΔF1, con remuestreos
  representados como pesos para evaluar cada réplica en O(n).
"""
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------
def midrank(x: np.ndarray) -> np.ndarray:
    """Rangos medios (1..n) con empates promediados, en O(n log n)."""
    return stats.rankdata(x, method="average")


def delong_fast(y: np.ndarray, scores: np.ndarray):
    """AUC y matriz de covarianza de DeLong para k clasificadores (Sun y Xu, 2014).

    y: etiquetas 0/1 de longitud n. scores: matriz (k, n) de puntuaciones continuas.
    Devuelve (auc de longitud k, covarianza k x k).
    """
    y = np.asarray(y).astype(bool)
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = midrank(pos[r])
        ty[r] = midrank(neg[r])
        tz[r] = midrank(np.concatenate([pos[r], neg[r]]))
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n          # componentes estructurales de los positivos
    v10 = 1.0 - (tz[:, m:] - ty) / m    # componentes estructurales de los negativos
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return auc, sx / m + sy / n


def delong_direct(y: np.ndarray, scores: np.ndarray):
    """Versión directa O(m·n) de DeLong (1988). Solo para validar `delong_fast` con n pequeño."""
    y = np.asarray(y).astype(bool)
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    k = scores.shape[0]
    m, n = y.sum(), (~y).sum()
    v10 = np.empty((k, m))
    v01 = np.empty((k, n))
    auc = np.empty(k)
    for r in range(k):
        X, Y = scores[r, y][:, None], scores[r, ~y][None, :]
        psi = (X > Y) + 0.5 * (X == Y)
        auc[r] = psi.mean()
        v10[r] = psi.mean(axis=1)
        v01[r] = psi.mean(axis=0)
    return auc, np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n


def delong_test(y, s1, s2, alpha: float = 0.05) -> dict:
    """Compara dos AUC correlacionados. Devuelve AUC, IC, ΔAUC con IC, z y p bilateral."""
    auc, cov = delong_fast(y, np.vstack([s1, s2]))
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = auc[0] - auc[1]
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var_diff, 0.0))
    z = diff / se if se > 0 else np.inf * np.sign(diff) if diff else 0.0
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else 0.0
    se1, se2 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return {
        "auc_1": auc[0], "auc_1_lo": auc[0] - zc * se1, "auc_1_hi": auc[0] + zc * se1,
        "auc_2": auc[1], "auc_2_lo": auc[1] - zc * se2, "auc_2_hi": auc[1] + zc * se2,
        "delta_auc": diff, "delta_lo": diff - zc * se, "delta_hi": diff + zc * se,
        "z": z, "p": p, "corr": cov[0, 1] / (se1 * se2) if se1 * se2 > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def mcnemar_test(y, pred1, pred2) -> dict:
    from statsmodels.stats.contingency_tables import mcnemar

    ok1, ok2 = np.asarray(pred1) == np.asarray(y), np.asarray(pred2) == np.asarray(y)
    b = int(np.sum(ok1 & ~ok2))   # acierta el modelo 1, falla el 2
    c = int(np.sum(~ok1 & ok2))   # falla el modelo 1, acierta el 2
    table = [[int(np.sum(ok1 & ok2)), b], [c, int(np.sum(~ok1 & ~ok2))]]
    exact = b + c < 25
    r = mcnemar(table, exact=exact, correction=True)
    return {"b": b, "c": c, "chi2": np.nan if exact else float(r.statistic), "p": float(r.pvalue),
            "exact": exact, "acc_1": ok1.mean(), "acc_2": ok2.mean()}


# ---------------------------------------------------------------------------
# Bootstrap pareado
# ---------------------------------------------------------------------------
class _WeightedMetrics:
    """Precalcula el orden de un vector de puntuaciones para evaluar AUC, AUC-PR y F1 con pesos en O(n)."""

    def __init__(self, y, score, threshold):
        order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
        s = np.asarray(score)[order]
        self.order = order
        self.y = np.asarray(y)[order].astype(np.float64)
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])   # inicio de cada grupo de empates
        self.pred = (np.asarray(score) >= threshold).astype(np.float64)

    def evaluate(self, w, y_orig):
        ws = w[self.order]
        P_g = np.add.reduceat(ws * self.y, self.starts)          # peso positivo por valor único (desc.)
        N_g = np.add.reduceat(ws * (1 - self.y), self.starts)
        P, N = P_g.sum(), N_g.sum()
        cumP, cumN = np.cumsum(P_g), np.cumsum(N_g)
        auc = np.sum(P_g * ((N - cumN) + 0.5 * N_g)) / (P * N)
        # Un grupo sin peso al inicio del ranking (no remuestreado) da 0/0; su P_g es 0 y no aporta a la suma.
        denom = cumP + cumN
        precision = np.divide(cumP, denom, out=np.zeros_like(cumP), where=denom > 0)
        ap = np.sum(P_g * precision) / P
        tp = np.sum(w * y_orig * self.pred)
        fp = np.sum(w * (1 - y_orig) * self.pred)
        fn = np.sum(w * y_orig * (1 - self.pred))
        f1 = 2 * tp / (2 * tp + fp + fn)
        return auc, ap, f1


def weighted_metrics(y, score, threshold, w=None):
    """AUC, AUC-PR (average precision) y F1 con pesos; con w=None equivale a los de scikit-learn."""
    y = np.asarray(y).astype(np.float64)
    w = np.ones_like(y) if w is None else w
    return _WeightedMetrics(y, score, threshold).evaluate(w, y)


def paired_bootstrap(y, s1, s2, thr1, thr2, B: int = 2000, seed: int = 42) -> dict:
    """Bootstrap pareado: mismos índices remuestreados para ambos modelos en cada réplica."""
    y = np.asarray(y).astype(np.float64)
    n = len(y)
    m1, m2 = _WeightedMetrics(y, s1, thr1), _WeightedMetrics(y, s2, thr2)
    rng = np.random.default_rng(seed)
    deltas = np.empty((B, 3))
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        deltas[b] = np.subtract(m1.evaluate(w, y), m2.evaluate(w, y))
    point = np.subtract(m1.evaluate(np.ones(n), y), m2.evaluate(np.ones(n), y))
    out = {}
    for j, k in enumerate(["auc", "pr_auc", "f1"]):
        d = deltas[:, j]
        lo, hi = np.percentile(d, [2.5, 97.5])
        # p bilateral por percentiles: proporción de réplicas al otro lado de 0 (acotada por 1/B).
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out[k] = {"delta": point[j], "boot_mean": d.mean(), "lo": lo, "hi": hi, "p": max(p, 1 / B)}
    return out


def holm(pvalues) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests

    return multipletests(np.asarray(pvalues, dtype=float), method="holm")[1]

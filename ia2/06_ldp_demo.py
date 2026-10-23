"""Module 3 (controlled demo): local differential privacy on the BLP quasi-identifiers."""
import json, numpy as np
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qi
rng = np.random.default_rng(42)
recs = qi.load_qi_records(); n0 = len(recs)
countries = [qi.country_of(r["parts"]) for r in recs]
top = [c for c,_ in Counter(countries).most_common(8)]
cat = np.array([top.index(c) if c in top else 8 for c in countries]); K = 9
years = np.array([r["y"] for r in recs], float)
EPS = [0.5, 1, 2, 4, 8]; TR = 500

def grr(vals, eps, K):
    p = np.exp(eps)/(np.exp(eps)+K-1); n = len(vals)
    keep = rng.random(n) < p
    other = (vals + rng.integers(1, K, n)) % K          # uniform over the K-1 other categories
    rep = np.where(keep, vals, other)
    q = (1-p)/(K-1); est = (np.bincount(rep, minlength=K)/n - q)/(p-q)
    return est
def central_laplace_hist(vals, eps, K):
    cnt = np.bincount(vals, minlength=K) + rng.laplace(0, 2/eps, K)   # replace-one sensitivity 2
    return cnt/len(vals)
def piecewise(x, eps):
    """Wang et al. 2019 Piecewise Mechanism, x in [-1,1]."""
    z = np.exp(eps/2); C = (z+1)/(z-1); n = len(x)
    l = (C+1)/2*x - (C-1)/2; r = l + C - 1
    inner = rng.random(n) < z/(z+1)
    u = rng.random(n)
    inner_v = l + u*(r-l)
    left_len, right_len = l + C, C - r
    t = rng.random(n)*(left_len+right_len)
    outer_v = np.where(t < left_len, -C + t, r + (t-left_len))
    return np.where(inner, inner_v, outer_v)
def laplace_num(x, eps): return x + rng.laplace(0, 2/eps, len(x))

def experiment(cat, years, tag):
    true_f = np.bincount(cat, minlength=K)/len(cat)
    x = (years - 1900)/ (2010-1900) * 2 - 1; tm = x.mean()
    out = {"n":len(cat), "eps":EPS, "grr_L1":[], "central_L1":[], "pm_mae_years":[], "lap_mae_years":[], "central_mean_mae_years":[]}
    for e in EPS:
        out["grr_L1"].append(float(np.mean([np.abs(grr(cat,e,K)-true_f).sum() for _ in range(TR)])))
        out["central_L1"].append(float(np.mean([np.abs(central_laplace_hist(cat,e,K)-true_f).sum() for _ in range(TR)])))
        sc = 55.0   # years per unit on [-1,1]
        out["pm_mae_years"].append(float(np.mean([abs(piecewise(x,e).mean()-tm) for _ in range(TR)])*sc))
        out["lap_mae_years"].append(float(np.mean([abs(laplace_num(x,e).mean()-tm) for _ in range(TR)])*sc))
        out["central_mean_mae_years"].append(float(np.mean([abs(tm + rng.laplace(0, 2/(len(x)*e)) - tm) for _ in range(TR)])*sc))
    return out
res = {"real_n299": experiment(cat, years, "real")}
big = rng.integers(0, n0, 10000)          # bootstrap to n=10,000 to show 1/sqrt(n) scaling
res["bootstrap_n10000"] = experiment(cat[big], years[big], "big")
res["categories"] = top + ["Other"]; res["true_freq"] = (np.bincount(cat, minlength=K)/n0).tolist()
json.dump(res, open("./results/ldp_results.json","w"), indent=1)
for k in ("real_n299","bootstrap_n10000"):
    print(k); [print(" ", kk, [round(v,3) for v in vv]) for kk, vv in res[k].items() if kk not in ("n","eps")]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
r = res["real_n299"]; b = res["bootstrap_n10000"]
ax[0].plot(EPS, r["grr_L1"], "o-", color="#c0392b", label="LDP randomised response, n=299")
ax[0].plot(EPS, b["grr_L1"], "s--", color="#c0392b", alpha=.55, label="LDP randomised response, n=10,000")
ax[0].plot(EPS, r["central_L1"], "o-", color="#1b6ca8", label="Central Laplace, n=299")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xlabel("ε"); ax[0].set_ylabel("L1 error of birth-country histogram"); ax[0].grid(alpha=.3); ax[0].legend(frameon=False, fontsize=6.5)
ax[1].plot(EPS, r["pm_mae_years"], "o-", color="#2e7d32", label="Piecewise mechanism (LDP), n=299")
ax[1].plot(EPS, r["lap_mae_years"], "o-", color="#c0392b", label="Laplace (LDP), n=299")
ax[1].plot(EPS, b["pm_mae_years"], "s--", color="#2e7d32", alpha=.55, label="Piecewise (LDP), n=10,000")
ax[1].plot(EPS, r["central_mean_mae_years"], "o-", color="#1b6ca8", label="Central Laplace, n=299")
ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xlabel("ε"); ax[1].set_ylabel("MAE of mean birth year (years)"); ax[1].grid(alpha=.3); ax[1].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig("./figures/fig_ldp.png", dpi=250)

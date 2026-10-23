"""Module 5 (controlled demo): compressive privacy -- PCA vs DCA vs DUCA projections of NER-model paragraph embeddings.
Utility U = birth continent (Europe / Americas / Asia / Africa+Oceania+Other); Privacy P = birth-decade group."""
import json, re, numpy as np, torch, scipy.linalg as sla
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qi
recs = qi.load_qi_records()
texts = {json.loads(l)["wiki_id"]: json.loads(l)["text"] for s in ("train","val","test") for l in open(f"{D}/ner_{s}.jsonl")}
recs = [r for r in recs if r["wiki_id"] in texts]; print("records:", len(recs))
cont = [qi.CONTINENT.get(qi.country_of(r["parts"]), "Other") for r in recs]
U = np.array([{"Europe":0,"Americas":1,"Asia":2}.get(c,3) for c in cont])
P = np.array([0 if r["y"]<1970 else 1 if r["y"]<1980 else 2 if r["y"]<1990 else 3 for r in recs])
# Representation: TF-IDF of the paragraph text (4-digit years generalised to decade tokens), reduced to 100 dims by SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
def prep(t): return re.sub(r"\b(1[89]\d|20\d)\d\b", lambda m: "DEC"+m.group(1), t)
docs = [prep(texts[r["wiki_id"]]) for r in recs]
tf = TfidfVectorizer(min_df=3, max_features=3000, sublinear_tf=True).fit_transform(docs)
X = TruncatedSVD(100, random_state=0).fit_transform(tf)
print("X", X.shape, "chance U(majority)=%.3f  P(majority)=%.3f" % (np.bincount(U).max()/len(U), np.bincount(P).max()/len(P)))

def scatter_between(Z, y):
    mu = Z.mean(0); S = np.zeros((Z.shape[1],)*2)
    for c in np.unique(y):
        d = (Z[y==c].mean(0)-mu)[:,None]; S += (y==c).sum()*d@d.T
    return S/len(Z)
def fit_projection(method, Z, u, p, m, rho=1.0):
    d = Z.shape[1]; reg = 1e-3*np.eye(d)
    if method == "PCA":
        w, V = np.linalg.eigh(np.cov(Z.T)); return V[:, ::-1][:, :m]
    Su = scatter_between(Z, u); St = np.cov(Z.T) + reg
    if method == "DCA":     # maximise utility signal / (total scatter as noise)
        w, V = sla.eigh(Su, St)
    else:                   # DUCA: maximise utility signal while penalising privacy-class scatter
        Sp = scatter_between(Z, p); w, V = sla.eigh(Su, St + rho*Sp*d/np.trace(Sp)*np.trace(St)/d)
    return V[:, ::-1][:, :m]
def evaluate(method, m, rho=1.0, seeds=(0,1,2)):
    au, ap = [], []
    for sd in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, U*4+P):
            sc = StandardScaler().fit(X[tr]); Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
            if method != "RAW":
                W = fit_projection(method, Ztr, U[tr], P[tr], m, rho); Ztr, Zte = Ztr@W, Zte@W
            s2 = StandardScaler().fit(Ztr); Ztr, Zte = s2.transform(Ztr), s2.transform(Zte)
            au.append(LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, U[tr]).score(Zte, U[te]))
            ap.append(LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, P[tr]).score(Zte, P[te]))
    return float(np.mean(au)), float(np.mean(ap)), float(np.std(au)), float(np.std(ap))
MS = [1, 2, 3, 4, 8, 16, 32, 64]
res = {"n":len(recs), "chance_U":float(np.bincount(U).max()/len(U)), "chance_P":float(np.bincount(P).max()/len(P)), "m":MS}
res["RAW"] = evaluate("RAW", 0)
for meth in ("PCA","DCA","DUCA"):
    res[meth] = [evaluate(meth, m) for m in MS]
    print(meth, [(round(a,3), round(b,3)) for a,b,_,_ in res[meth]])
print("RAW", res["RAW"])
json.dump(res, open(f"{ROOT}/results/compressive_privacy.json","w"), indent=1)
fig, ax = plt.subplots(1,2, figsize=(7.4,3.0), sharey=False)
for meth, c in (("PCA","#7f8c8d"),("DCA","#1b6ca8"),("DUCA","#c0392b")):
    ax[0].plot(MS, [r[0] for r in res[meth]], "o-", color=c, label=meth); ax[1].plot(MS, [r[1] for r in res[meth]], "o-", color=c, label=meth)
for a, k, t in ((ax[0],"chance_U","Utility: birth continent (accuracy)"),(ax[1],"chance_P","Privacy leak: birth decade (adversary accuracy)")):
    a.axhline(res[k], color="k", ls=":", lw=1, label="majority baseline"); a.axhline(res["RAW"][0 if k=="chance_U" else 1], color="g", ls="--", lw=1, label="uncompressed")
    a.set_xscale("log", base=2); a.set_xlabel("Projected dimensions m"); a.set_title(t, fontsize=8); a.grid(alpha=.3)
ax[0].legend(frameon=False, fontsize=6.5); fig.tight_layout(); fig.savefig(f"{ROOT}/figures/fig_cp.png", dpi=250)

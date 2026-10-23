import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="./results"; dp=json.load(open(f"{R}/dpsgd_results.json")); b=json.load(open(f"{R}/baseline_ia2.json"))["test"]["f1"]; be=json.load(open(f"{R}/baseline_ia2_es.json"))["test"]["f1"]
fig, ax = plt.subplots(figsize=(5.2,3.3))
xs=[4,16,64]; ax.plot(xs,[dp[str(x)]["test"]["f1"] for x in xs],"o-",color="#c0392b",label="DP-SGD, 15 epochs (90 steps)")
if "16_e40" in dp: ax.plot([16],[dp["16_e40"]["test"]["f1"]],"s",color="#1b6ca8",ms=7,label="DP-SGD, 40 epochs (240 steps)")
ax.axhline(be,color="k",ls="--",lw=1,label="Non-private, early stopped"); ax.axhline(b,color="gray",ls=":",lw=1.2,label="Non-private, final epoch")
ax.set_xscale("log",base=2); ax.set_xticks(xs); ax.set_xticklabels(["4","16","64"]); ax.set_xlabel("Privacy budget ε (δ = 1e-3)"); ax.set_ylabel("Test entity-level F1"); ax.set_ylim(0,0.32); ax.grid(alpha=.3); ax.legend(frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig("./figures/fig_dp.png",dpi=250)

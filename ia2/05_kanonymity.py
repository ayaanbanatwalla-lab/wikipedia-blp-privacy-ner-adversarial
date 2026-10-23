"""Module 4: k-anonymity on the quasi-identifiers (birth date, birth place) of the BLPs in the corpus."""
import json, re, itertools, csv
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "."
raw = [json.loads(l) for l in open(f"{ROOT}/data/raw_blp_data.jsonl", encoding="utf-8")]
def clean_part(p):
    p = re.sub(r"\{\{.*", "", p); p = re.sub(r"\[https?:[^\]]*\]?", "", p); p = re.sub(r"\(.*?\)|\)", "", p)
    return p.strip()
CONTINENT = {}
for c in ["Norway","Finland","Sweden","Denmark","Netherlands","Spain","France","Italy","Russia","Romania","Estonia","West Germany","Faroe Islands","Soviet Union","Uzbek SSR","United Kingdom"]: CONTINENT[c] = "Europe"
for c in ["India","Nepal","Myanmar","Taiwan","Saudi Arabia","Philippines","Vietnam","China","Afghanistan","Bahrain","Pakistan","Republic of China","Malaysia","Iran","Federation of Malaya","British India"]: CONTINENT[c] = "Asia"
for c in ["Morocco","Nigeria","South Africa","Egypt","Algeria","Liberia","Cameroon","Tanzania","Togo","Chad","Ghana","Kenya"]: CONTINENT[c] = "Africa"
for c in ["United States","Canada","Argentina","Brazil","Jamaica","Mexico","Ecuador","Chile","Dominican Republic","Netherlands Antilles","American Samoa","Guam"]: CONTINENT[c] = "Americas"
for c in ["Australia","New Zealand","Papua New Guinea"]: CONTINENT[c] = "Oceania"
US_STATES = {"California","Louisiana","Florida","Texas","New York City","New York","Georgia","Colorado","Oklahoma","Kentucky"}
CITY_COUNTRY = {"Bergen":"Norway","Bærum":"Norway","Helsinki":"Finland","Lahti":"Finland","Tartu":"Estonia","Emmen":"Netherlands","North Brabant":"Netherlands","Delhi":"India","Chittorgarh":"India"}
COUNTRY = {"U.S.":"United States","US":"United States","USA":"United States","United States":"United States","U.S":"United States",
           "England":"United Kingdom","Scotland":"United Kingdom","Wales":"United Kingdom","Northern Ireland":"United Kingdom","UK":"United Kingdom","U.K.":"United Kingdom"}
def date_g(y, mo, level):  # 0 full(not distinguishable from month here: day dropped at L1) ...
    return [f"{y}-{mo:02d}-DD*", f"{y}-{mo:02d}", f"{y}", f"{y//5*5}-{y//5*5+4}", f"{y//10*10}s", "*"][level]
def country_of(parts):
    c = parts[-1]
    if c in US_STATES: return "United States"
    if c in CITY_COUNTRY: return CITY_COUNTRY[c]
    return COUNTRY.get(c, c)
def place_g(parts, level):
    if level == 0: return ", ".join(parts)
    if level == 1: return ", ".join(parts[1:]) if len(parts) > 1 else parts[0]
    if level == 2: return country_of(parts)
    if level == 3: return CONTINENT.get(country_of(parts), "Other")
    return "*"
recs = []
for r in raw:
    bd, bp = r["infobox"].get("birth_date",""), r["infobox"].get("birth_place","")
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", bd or "")
    parts = [q for q in (clean_part(p) for p in re.sub(r"\[\[|\]\]","",bp).split(",")) if q]
    if m and parts: recs.append((int(m.group(1)), int(m.group(2)), bd, parts))
N = len(recs); print("records with both QIs:", N)
# NOTE: level 0 for date is the exact DOB; the day is stored implicitly -> use full ISO at L0
def key(i, dl, pl):
    y, mo, iso, parts = recs[i]
    d = iso if dl == 0 else date_g(y, mo, dl)
    return (d, place_g(parts, pl))
DMAX, PMAX = 5, 4
rows = []
for dl, pl in itertools.product(range(DMAX+1), range(PMAX+1)):
    cnt = Counter(key(i, dl, pl) for i in range(N))
    sizes = [cnt[key(i, dl, pl)] for i in range(N)]
    k = min(sizes)
    unique = sum(s == 1 for s in sizes)/N
    loss = (dl/DMAX + pl/PMAX)/2   # normalised generalisation height (Samarati precision-loss)
    row = {"date_level":dl,"place_level":pl,"min_k":k,"pct_unique":unique,"info_loss":loss,"classes":len(cnt)}
    for K in (2,3,5,10): row[f"suppress_for_k{K}"] = sum(s < K for s in sizes)/N
    rows.append(row)
with open(f"{ROOT}/results/kanonymity_lattice.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
best = {}
for K in (2,3,5,10):
    ok = [r for r in rows if r[f"suppress_for_k{K}"] <= 0.05]
    b = min(ok, key=lambda r: (r["info_loss"], r["date_level"]+r["place_level"]))
    best[K] = b; print(f"k={K}: date L{b['date_level']}, place L{b['place_level']}, loss={b['info_loss']:.2f}, suppressed={b[f'suppress_for_k{K}']:.3f}, min_k_after={b['min_k']}")
json.dump({"n":N,"best":{str(k):v for k,v in best.items()},"raw_unique_pct":rows[0]["pct_unique"]}, open(f"{ROOT}/results/kanonymity_summary.json","w"), indent=1)
print("L0,L0 unique:", rows[0]["pct_unique"])
# figure: privacy (records in classes < k=5) vs utility loss
fig, ax = plt.subplots(figsize=(5.2,3.4))
for K, c in zip((2,5,10), ("#1b6ca8","#c0392b","#2e7d32")):
    xs = [r["info_loss"] for r in rows]; ys = [1-r[f"suppress_for_k{K}"] for r in rows]
    ax.scatter(xs, ys, s=22, color=c, label=f"k={K}", alpha=.8)
ax.set_xlabel("Information loss (normalised generalisation height)"); ax.set_ylabel("Fraction of records with class size ≥ k")
ax.grid(alpha=.3); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{ROOT}/figures/fig_kanon.png", dpi=250)

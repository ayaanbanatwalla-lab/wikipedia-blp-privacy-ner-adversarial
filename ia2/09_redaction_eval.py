"""Module 4: NER-based redaction leakage under adversarial context perturbation, per defended model."""
import sys
from attack import *
sp = load_splits(); tok = get_tok(); names = sys.argv[1:] or ["ia2_baseline","ia2_adv_trained"]
PII = set(["PERSON"] + PRIVACY_LABELS)
out_path = f"{ROOT}/results/redaction_eval.json"
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
def gold_miss(model, records, labels):
    """clean redaction miss rate: gold entities of the given labels not covered (>=50% overlap) by any predicted span."""
    att = NERAdversarialAttack(model, tok); miss = tot = 0
    for r in records:
        preds = att.predict_spans(r["text"])
        for g in r["entities"]:
            if g["label"] in labels:
                tot += 1; miss += (not span_overlap_any(g, preds))
    return miss, tot
for nm in names:
    model = load_model(f"{D}/{nm}.pth")
    inst = run_attack(model, tok, sp["test"])
    slim = [{"wiki_id":r["wiki_id"],"label":r["target_entity"]["label"],"flip":r["success"],"red_clean":r["redacted_clean"],"red_adv":r["redacted_adv"],
             "conf_drop":r["clean_confidence"]-r["adversarial_confidence"],"mods":r["num_modifications"]} for r in inst]
    def agg(sub):
        n = len(sub)
        return {"n":n,"flip_rate":sum(x["flip"] for x in sub)/max(n,1),"extra_leak_rate":sum(x["red_clean"] and not x["red_adv"] for x in sub)/max(n,1),
                "avg_conf_drop":sum(x["conf_drop"] for x in sub)/max(n,1)}
    pii = [x for x in slim if x["label"] in PII]; gen = [x for x in slim if x["label"] not in PII]
    mi, to = gold_miss(model, sp["test"], PII); mi_all, to_all = gold_miss(model, sp["test"], set(ALL_ENTITY_TYPES))
    res[nm] = {"all":agg(slim),"pii":agg(pii),"general":agg(gen),"clean_miss_pii":[mi,to],"clean_miss_all":[mi_all,to_all],
               "by_label":{l:agg([x for x in slim if x["label"]==l]) for l in sorted(set(x["label"] for x in slim))}}
    json.dump(res, open(out_path,"w"), indent=1); print(nm, res[nm]["all"], res[nm]["pii"], flush=True)

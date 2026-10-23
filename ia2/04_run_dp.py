"""Module 2: DP-SGD epsilon sweep -> utility (clean F1) + robustness (attack) per epsilon."""
import sys
from dp import *
from attack import run_attack, summarize
EPS = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 4, 16, 64]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-3
BS = int(sys.argv[4]) if len(sys.argv) > 4 else 64
TAG = sys.argv[5] if len(sys.argv) > 5 else ""
sp = load_splits(); tok = get_tok()
te = torch.utils.data.DataLoader(NERDataset(sp["test"], tok), batch_size=32)
out_path = f"{ROOT}/results/dpsgd_results.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}
for eps in EPS:
    if eps == 0:   # non-private control with identical optimiser settings (no clipping / noise)
        torch.manual_seed(SEED); random.seed(SEED)
        model = BertForTokenClassification(make_config()); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        trl = torch.utils.data.DataLoader(NERDataset(sp["train"], tok), batch_size=BS, shuffle=True)
        vl = torch.utils.data.DataLoader(NERDataset(sp["val"], tok), batch_size=32); hist = []
        for ep in range(1, EPOCHS+1):
            model.train()
            for b in trl:
                opt.zero_grad(); model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss.backward(); opt.step()
            hist.append({"epoch":ep,"val_f1":entity_level_prf(model, vl)[0]["f1"]}); print("control", ep, hist[-1]["val_f1"], flush=True)
        info = {"target_eps":None,"history":hist,"sigma":0.0,"clip":None,"epochs":EPOCHS,"delta":None,"eps_spent":None}
    else:
        model, info = train_dp(eps, epochs=EPOCHS, lr=LR, bs=BS, log=lambda s: print(s, flush=True))
    torch.save(model.state_dict(), f"{D}/ia2_dp_eps{eps:g}{TAG}.pth")
    t, pt = entity_level_prf(model, te)
    att = run_attack(model, tok, sp["test"]); s = summarize(att)
    info.update({"lr": LR, "batch_size": BS, "test": t, "attack": s}); results[f"{eps:g}{TAG}"] = info
    json.dump(results, open(out_path, "w"), indent=1)
    print("DONE", eps, t["f1"], s, flush=True)

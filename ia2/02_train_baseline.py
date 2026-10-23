"""Re-establish the IA-2 baseline (same architecture/hyper-parameters as IA-1: AdamW 5e-4, wd 0.01, cosine, 25 ep, bs 16, seed 42)."""
from common import *
import time
torch.manual_seed(SEED); random.seed(SEED)
sp = load_splits(); tok = get_tok()
tr = torch.utils.data.DataLoader(NERDataset(sp["train"], tok), batch_size=16, shuffle=True)
va = torch.utils.data.DataLoader(NERDataset(sp["val"], tok), batch_size=16)
te = torch.utils.data.DataLoader(NERDataset(sp["test"], tok), batch_size=16)
model = BertForTokenClassification(make_config())
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
EP = 25; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
hist = []; t0 = time.time()
for ep in range(1, EP+1):
    model.train(); tl = n = 0
    for b in tr:
        opt.zero_grad(); out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
        out.loss.backward(); opt.step(); tl += out.loss.item(); n += 1
    sched.step(); v,_ = entity_level_prf(model, va)
    hist.append({"epoch":ep,"train_loss":tl/n,"val_f1":v["f1"],"val_precision":v["precision"],"val_recall":v["recall"]})
    print(ep, round(tl/n,4), round(v["f1"],4), f"{time.time()-t0:.0f}s", flush=True)
torch.save(model.state_dict(), f"{D}/ia2_baseline.pth")
t,pt = entity_level_prf(model, te); print("TEST", t)
json.dump({"history":hist,"test":t,"per_type":pt}, open(f"{ROOT}/results/baseline_ia2.json","w"), indent=1)

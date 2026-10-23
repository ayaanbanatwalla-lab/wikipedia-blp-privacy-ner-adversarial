"""Module 2 ablation controls (same optimiser, batch size, epochs, loss as the DP runs): (A) clip-only (sigma=0), (B) no clipping, no noise."""
from dp import *
sp = load_splits(); tok = get_tok()
te = torch.utils.data.DataLoader(NERDataset(sp["test"], tok), batch_size=32); res = {}
m, info = train_dp(0, epochs=15, lr=3e-3, bs=64, noise=0.0, log=lambda s: print(s, flush=True))
t,_ = entity_level_prf(m, te); res["clip_only"] = {"test": t, "history": info["history"]}; print("clip_only", t, flush=True)
torch.manual_seed(SEED); random.seed(SEED)
model = BertForTokenClassification(make_config()); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
trl = torch.utils.data.DataLoader(NERDataset(sp["train"], tok), batch_size=64, shuffle=True)
vl = torch.utils.data.DataLoader(NERDataset(sp["val"], tok), batch_size=32); hist = []
for ep in range(1, 16):
    model.train()
    for b in trl:
        opt.zero_grad(); per_sample_loss(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits, b["labels"]).mean().backward(); opt.step()
    hist.append({"epoch":ep,"val_f1":entity_level_prf(model, vl)[0]["f1"]}); print("noclip", ep, hist[-1]["val_f1"], flush=True)
t,_ = entity_level_prf(model, te); res["no_clip_no_noise"] = {"test": t, "history": hist}; print("no_clip_no_noise", t, flush=True)
json.dump(res, open(f"{ROOT}/results/dp_controls.json","w"), indent=1)

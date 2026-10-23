from common import *
import torch.nn.functional as F, time
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def per_sample_loss(logits, labels):
    ce = F.cross_entropy(logits.transpose(1,2), labels, ignore_index=-100, reduction="none")
    m = (labels != -100).float()
    return (ce*m).sum(1)/m.sum(1).clamp(min=1)

def train_dp(eps, epochs=15, delta=1e-3, clip=1.0, lr=5e-4, seed=SEED, bs=16, log=print, wd=0.01, noise=None):
    torch.manual_seed(seed); random.seed(seed)
    sp = load_splits(); tok = get_tok()
    tr = torch.utils.data.DataLoader(NERDataset(sp["train"], tok), batch_size=bs)
    va = torch.utils.data.DataLoader(NERDataset(sp["val"], tok), batch_size=32)
    model = BertForTokenClassification(make_config())
    assert not ModuleValidator.validate(model, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pe = PrivacyEngine(accountant="rdp")
    if noise is None:
        model, opt, tr = pe.make_private_with_epsilon(module=model, optimizer=opt, data_loader=tr, epochs=epochs,
            target_epsilon=eps, target_delta=delta, max_grad_norm=clip, poisson_sampling=True)
    else:   # ablation: fixed noise multiplier (0 = clipping only)
        model, opt, tr = pe.make_private(module=model, optimizer=opt, data_loader=tr, noise_multiplier=noise, max_grad_norm=clip, poisson_sampling=True)
    sigma = opt.noise_multiplier; hist = []; t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); tl = n = 0
        for b in tr:
            if b["input_ids"].shape[0] == 0: continue
            opt.zero_grad()
            B, T = b["input_ids"].shape  # explicit batch-sized position/type ids -> per-sample grads for embeddings
            pos = torch.arange(T).unsqueeze(0).expand(B, T).contiguous(); tt = torch.zeros(B, T, dtype=torch.long)
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=pos, token_type_ids=tt).logits
            loss = per_sample_loss(logits, b["labels"]).mean(); loss.backward(); opt.step()
            tl += loss.item(); n += 1
        v,_ = entity_level_prf(model, va)
        hist.append({"epoch":ep,"loss":tl/n,"val_f1":v["f1"],"eps_spent":(pe.get_epsilon(delta) if noise != 0 else float("inf"))})
        log(f"eps_target={eps} ep{ep} loss={tl/n:.3f} valF1={v['f1']:.4f} {time.time()-t0:.0f}s")
    plain = BertForTokenClassification(make_config())
    plain.load_state_dict({k.replace("_module.",""):v for k,v in model.state_dict().items()}); plain.eval()
    return plain, {"target_eps":eps,"delta":delta,"sigma":sigma,"clip":clip,"epochs":epochs,"eps_spent":(pe.get_epsilon(delta) if noise != 0 else float("inf")),
                   "sample_rate":1/len(tr),"history":hist}

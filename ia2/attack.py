"""NER-adapted black-box attack (identical logic to IA-1 notebook Part F)."""
from common import *
import random as _random
from nltk.corpus import wordnet as wn

nlp_pos = spacy.load("en_core_web_sm", disable=["ner"])

def get_synonyms(word, pos_tag):
    """WordNet-based candidate substitutions, POS-filtered to preserve grammaticality."""
    wn_pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
    wn_pos = wn_pos_map.get(pos_tag)
    if wn_pos is None:
        return []
    syns = set()
    for syn in wn.synsets(word.lower(), pos=wn_pos):
        for lemma in syn.lemmas():
            cand = lemma.name().replace("_", " ")
            if cand.lower() != word.lower() and cand.isalpha():
                syns.add(cand)
    return list(syns)[:8]

class NERAdversarialAttack:
    """NER-adapted, constrained synonym-substitution adversarial attack, inspired by
    TextFooler (Jin et al., 2020) and BERT-Attack (Li et al., 2020), but ADAPTED for
    token-classification NER rather than the original text-classification setting:
      - The attack targets CORRECTLY-predicted privacy-sensitive entities.
      - Only CONTEXT words (outside any entity span) are perturbed -- entity tokens
        (including the person's own name) are never modified, so the identity-bearing
        information stays intact and degradation is attributable to context manipulation.
      - Word importance is measured via leave-one-out: how much removing/masking a context
        word changes the target entity's predicted-label CONFIDENCE (continuous signal).
      - Candidate substitutions are POS-filtered WordNet synonyms (grammaticality constraint).
    Threat model: QUERY-BASED BLACK-BOX -- only needs model output probabilities, never
    touches model weights, training data, or architecture.
    """
    def __init__(self, model, tokenizer, max_perturbations=3, max_queries=60):
        self.model = model
        self.tokenizer = tokenizer
        self.max_perturbations = max_perturbations
        self.max_queries = max_queries

    def predict_spans(self, text):
        tokens, attn, _ = align_labels(text, [], self.tokenizer)
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([tokens]),
                                 attention_mask=torch.tensor([attn])).logits[0]
        pred_ids = logits.argmax(-1).tolist()
        enc = self.tokenizer.encode(text)
        offsets = enc.offsets
        spans, cur_type, cur_start_char, prev_end = [], None, None, 0
        for i, lid in enumerate(pred_ids):
            tok_start, tok_end = offsets[i]
            if tok_start == tok_end == 0:
                if cur_type is not None:
                    spans.append({"start": cur_start_char, "end": prev_end, "label": cur_type})
                    cur_type = None
                continue
            label = ID2LABEL[lid]
            if label == "O":
                if cur_type is not None:
                    spans.append({"start": cur_start_char, "end": prev_end, "label": cur_type})
                    cur_type = None
            elif label.startswith("B-"):
                if cur_type is not None:
                    spans.append({"start": cur_start_char, "end": prev_end, "label": cur_type})
                cur_type = label[2:]; cur_start_char = tok_start
            elif label.startswith("I-") and cur_type == label[2:]:
                pass
            elif label.startswith("I-"):
                if cur_type is not None:
                    spans.append({"start": cur_start_char, "end": prev_end, "label": cur_type})
                cur_type = label[2:]; cur_start_char = tok_start
            prev_end = tok_end
        if cur_type is not None:
            spans.append({"start": cur_start_char, "end": prev_end, "label": cur_type})
        return spans

    def get_target_confidence(self, text, target):
        """Softmax probability of the correct B-<type> label at the target entity's first
        token -- a continuous degradation signal, not just a binary flip check."""
        tokens, attn, _ = align_labels(text, [], self.tokenizer)
        enc = self.tokenizer.encode(text)
        offsets = enc.offsets
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([tokens]),
                                 attention_mask=torch.tensor([attn])).logits[0]
        probs = torch.softmax(logits, dim=-1)
        target_label_id = LABEL2ID.get(f"B-{target['label']}")
        if target_label_id is None:
            return 0.0
        for i, (s, e) in enumerate(offsets):
            if s == target["start"] and e > s:
                return probs[i, target_label_id].item()
        return 0.0

    def attack(self, text, gold_entities, target_labels=None):
        """Attack a paragraph: every correctly-detected qualifying entity is attacked in
        turn. Returns a list of result dicts (possibly empty)."""
        doc = nlp_pos(text)
        clean_preds = self.predict_spans(text)
        def spans_match(a, b):
            return a["start"]==b["start"] and a["end"]==b["end"] and a["label"]==b["label"]
        correctly_detected = [g for g in gold_entities if any(spans_match(g,p) for p in clean_preds)
                               if target_labels is None or g["label"] in target_labels]
        if not correctly_detected:
            return []
        return [r for r in (self._attack_one_entity(text, gold_entities, t, doc)
                             for t in correctly_detected) if r is not None]

    def _attack_one_entity(self, text, gold_entities, target, doc):
        clean_preds = self.predict_spans(text)
        protected_chars = set()
        for g in gold_entities:
            protected_chars.update(range(g["start"], g["end"]))
        candidates = [t for t in doc if t.is_alpha and len(t.text) >= 3
                      and not any(c in protected_chars for c in range(t.idx, t.idx+len(t.text)))
                      and t.pos_ in ("NOUN","VERB","ADJ","ADV")]

        queries = 0
        clean_confidence = self.get_target_confidence(text, target)
        def confidence_drop(txt):
            nonlocal queries
            queries += 1
            return clean_confidence - self.get_target_confidence(txt, target)
        def target_still_detected(txt):
            preds = self.predict_spans(txt)
            return any(p["label"]==target["label"] and abs(p["start"]-target["start"])<=15 for p in preds)

        scored = []
        for t in candidates:
            if queries >= self.max_queries: break
            masked = text[:t.idx] + ("x"*len(t.text)) + text[t.idx+len(t.text):]
            scored.append((t, confidence_drop(masked)))
        scored.sort(key=lambda x: -x[1])

        adv_text, modified_tokens, offset_shift = text, [], 0
        for t, importance in scored:
            if len(modified_tokens) >= self.max_perturbations or queries >= self.max_queries:
                break
            syns = get_synonyms(t.text, t.pos_)
            _random.shuffle(syns)
            best_cand, best_drop = None, -1.0
            for cand in syns:
                if queries >= self.max_queries: break
                trial_start = t.idx + offset_shift
                trial_text = adv_text[:trial_start] + cand + adv_text[trial_start+len(t.text):]
                drop = confidence_drop(trial_text)
                if drop > best_drop:
                    best_drop, best_cand = drop, cand
            if best_cand is not None and best_drop > 0:
                trial_start = t.idx + offset_shift
                adv_text = adv_text[:trial_start] + best_cand + adv_text[trial_start+len(t.text):]
                offset_shift += len(best_cand) - len(t.text)
                modified_tokens.append({"original": t.text, "replacement": best_cand, "confidence_drop": best_drop, "idx": t.idx, "delta": len(best_cand)-len(t.text)})

        adv_preds = self.predict_spans(adv_text)
        success = not target_still_detected(adv_text)
        return {"original_text": text, "adversarial_text": adv_text, "target_entity": target,
                "clean_predictions": clean_preds, "adversarial_predictions": adv_preds,
                "clean_confidence": clean_confidence,
                "adversarial_confidence": self.get_target_confidence(adv_text, target),
                "modified_tokens": modified_tokens, "num_modifications": len(modified_tokens),
                "num_queries": queries, "success": success}


import time
def span_overlap_any(target, preds):
    """Type-agnostic: is the target span still covered (>=50% char overlap) by ANY predicted entity (i.e. would be redacted)?"""
    ts, te = target["start"], target["end"]
    for p in preds:
        ov = min(te, p["end"]) - max(ts, p["start"])
        if ov >= 0.5*(te-ts): return True
    return False

def run_attack(model, tok, records, seed=SEED, max_pert=3, max_q=60, targets=None, limit=None):
    """Attack every correctly-detected entity (optionally restricted to label set `targets`). Returns per-instance dicts."""
    _random.seed(seed)
    att = NERAdversarialAttack(model, tok, max_perturbations=max_pert, max_queries=max_q)
    out = []
    for r in (records[:limit] if limit else records):
        for res in att.attack(r["text"], r["entities"], target_labels=targets):
            res["wiki_id"] = r["wiki_id"]
            tg = res["target_entity"]
            shift = sum(m["delta"] for m in res["modified_tokens"] if m["idx"] < tg["start"])
            tg_adv = {"start": tg["start"]+shift, "end": tg["end"]+shift}
            res["redacted_clean"] = span_overlap_any(tg, res["clean_predictions"])
            res["redacted_adv"] = span_overlap_any(tg_adv, res["adversarial_predictions"])
            out.append(res)
    return out

def summarize(res):
    n = len(res)
    if n == 0: return {"n":0}
    return {"n":n, "flip_rate": sum(r["success"] for r in res)/n,
            "leak_rate_any_type": sum((r["redacted_clean"] and not r["redacted_adv"]) for r in res)/n,
            "avg_conf_drop": sum(r["clean_confidence"]-r["adversarial_confidence"] for r in res)/n,
            "avg_mods": sum(r["num_modifications"] for r in res)/n,
            "avg_queries": sum(r["num_queries"] for r in res)/n}

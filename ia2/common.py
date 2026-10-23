"""Shared pipeline for IA-2 experiments (identical logic to the IA-1 notebook)."""
import json, re, random, os, csv
from collections import Counter
import torch, spacy
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertForTokenClassification

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")
SEED = 42; MAX_LEN = 128
GENERAL_LABELS = ["PERSON", "ORG", "LOC", "DATE"]
PRIVACY_LABELS = ["EMAIL","PHONE","ADDRESS","FAMILY","HEALTH","RELIGION","NATIONALITY","DATE_OF_BIRTH",
                  "EDUCATION","EMPLOYMENT","SOCIAL_MEDIA","IDENTIFIER","FINANCIAL","LEGAL","OTHER_SENSITIVE"]
ALL_ENTITY_TYPES = GENERAL_LABELS + PRIVACY_LABELS
LABEL_LIST = ["O"]
for e in ALL_ENTITY_TYPES: LABEL_LIST += [f"B-{e}", f"I-{e}"]
LABEL2ID = {l:i for i,l in enumerate(LABEL_LIST)}; ID2LABEL = {i:l for l,i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
SOCIAL_RE = re.compile(r"(?:@[A-Za-z0-9_]{3,})|(?:(?:twitter|instagram|facebook)\.com/[A-Za-z0-9_.]+)")
IDENTIFIER_RE = re.compile(r"\b(?:ID|SSN|passport)\s*[:#]?\s*[A-Z0-9]{5,}\b", re.IGNORECASE)
HEALTH_KW = ["cancer","diagnosed","disability","disabled","illness","syndrome","disorder","depression","autism","paralysis","blind","deaf","hiv","aids"]
LEGAL_KW = ["convicted","lawsuit","sentenced","arrested","indicted","acquitted","prison","felony","misdemeanor","court case","sued"]
FINANCIAL_KW = ["net worth","salary","earned","fortune","million dollars","billion dollars"]
SPACY_MAP = {"PERSON":"PERSON","ORG":"ORG","GPE":"LOC","LOC":"LOC","DATE":"DATE"}
PRIORITY = set(PRIVACY_LABELS)

def find_keyword_spans(text, keywords, label):
    spans, low = [], text.lower()
    for kw in keywords:
        start = 0
        while (idx := low.find(kw, start)) != -1:
            spans.append({"start": idx, "end": idx+len(kw), "label": label}); start = idx+len(kw)
    return spans
def find_regex_spans(text, pattern, label):
    return [{"start": m.start(), "end": m.end(), "label": label} for m in pattern.finditer(text)]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
def date_variants(val):
    """IA-2 refinement: infobox dates are ISO (1964-02-22) but article text writes '22 February 1964' or
    'February 22, 1964'. Generate textual variants so DATE_OF_BIRTH can be cross-referenced."""
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$", val or "")
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12:
            return [f"{d} {MONTHS[mo-1]} {y}", f"{MONTHS[mo-1]} {d}, {y}"]
    return [val.strip()] if val and len(val.strip()) >= 3 else []

def find_infobox_spans(text, infobox):
    spans, low = [], text.lower()
    def try_match(fv, label):
        if not fv or len(fv.strip()) < 3: return
        idx = low.find(fv.lower().strip())
        if idx != -1: spans.append({"start": idx, "end": idx+len(fv.strip()), "label": label})
    if "birth_date" in infobox:
        for variant in date_variants(infobox["birth_date"]):
            idx = low.find(variant.lower())
            if idx != -1:
                spans.append({"start": idx, "end": idx+len(variant), "label": "DATE_OF_BIRTH"}); break
    if "religion" in infobox: try_match(infobox["religion"], "RELIGION")
    if "nationality" in infobox: try_match(infobox["nationality"], "NATIONALITY")
    for f in ["spouse","children","parents","partner"]:
        if f in infobox: try_match(infobox[f], "FAMILY")
    for f in ["alma_mater","education"]:
        if f in infobox: try_match(infobox[f], "EDUCATION")
    for f in ["occupation","employer"]:
        if f in infobox: try_match(infobox[f], "EMPLOYMENT")
    return spans
def remove_overlaps(spans):
    spans = sorted(spans, key=lambda s: (-(s["end"]-s["start"]), s["label"] not in PRIORITY))
    kept_spans, occ = [], [False]*(max((s["end"] for s in spans), default=0))
    for s in spans:
        if any(occ[s["start"]:s["end"]]): continue
        kept_spans.append(s)
        for i in range(s["start"], s["end"]): occ[i] = True
    return sorted(kept_spans, key=lambda s: s["start"])

def build_data():
    """raw_blp_data.jsonl -> clean corpus -> silver annotation -> 70/15/15 split -> tokenizer."""
    kept = [json.loads(l) for l in open(f"{D}/raw_blp_data.jsonl", encoding="utf-8")]
    random.seed(SEED); random.shuffle(kept)
    clean = [{"wiki_id": r["wiki_id"], "title": re.sub(r"\s+"," ",r["title"]).strip(), "url": r["url"],
              "paragraph": re.sub(r"\s+"," ",r["paragraph"]).strip(), "infobox": r["infobox"]} for r in kept[:500]]
    nlp_full = spacy.load("en_core_web_sm")
    annotated = []
    for r, doc in zip(clean, nlp_full.pipe([r["paragraph"] for r in clean], batch_size=64)):
        text = r["paragraph"]
        spans = (find_regex_spans(text, EMAIL_RE, "EMAIL") + find_regex_spans(text, SOCIAL_RE, "SOCIAL_MEDIA")
                 + find_regex_spans(text, IDENTIFIER_RE, "IDENTIFIER") + find_keyword_spans(text, HEALTH_KW, "HEALTH")
                 + find_keyword_spans(text, LEGAL_KW, "LEGAL") + find_keyword_spans(text, FINANCIAL_KW, "FINANCIAL")
                 + find_infobox_spans(text, r["infobox"]))
        for ent in doc.ents:
            if ent.label_ in SPACY_MAP:
                spans.append({"start": ent.start_char, "end": ent.end_char, "label": SPACY_MAP[ent.label_]})
        annotated.append({"wiki_id": r["wiki_id"], "title": r["title"], "text": text, "infobox": r["infobox"],
                          "entities": remove_overlaps(spans), "annotation_source": "silver_standard_weak_supervision",
                          "human_verified": False})
    # NOTE: WordPiece training is not bit-reproducible across runs, so an existing vocabulary file is reused
    # (the checkpoints were trained with it). Delete data/privbert_tokenizer-vocab.txt to retrain it.
    if not os.path.exists(f"{D}/privbert_tokenizer-vocab.txt"):
        with open(f"{D}/tokenizer_train_corpus.txt","w",encoding="utf-8") as f:
            for r in clean: f.write(r["paragraph"]+"\n")
        t = BertWordPieceTokenizer(lowercase=False, strip_accents=False)
        t.train(files=[f"{D}/tokenizer_train_corpus.txt"], vocab_size=12000, min_frequency=2,
                special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"])
        t.save_model(D, "privbert_tokenizer")
    random.seed(SEED)
    idx = list(range(len(annotated))); random.shuffle(idx)
    n = len(idx); ntr = int(0.70*n); nva = int(0.15*n)
    splits = {"train":[annotated[i] for i in idx[:ntr]], "val":[annotated[i] for i in idx[ntr:ntr+nva]],
              "test":[annotated[i] for i in idx[ntr+nva:]]}
    for k,v in splits.items():
        with open(f"{D}/ner_{k}.jsonl","w",encoding="utf-8") as f:
            for d in v: f.write(json.dumps(d, ensure_ascii=False)+"\n")
    return splits

def load_splits():
    return {k:[json.loads(l) for l in open(f"{D}/ner_{k}.jsonl",encoding="utf-8")] for k in ["train","val","test"]}

def get_tok():
    tok = BertWordPieceTokenizer(f"{D}/privbert_tokenizer-vocab.txt", lowercase=False, strip_accents=False)
    tok.enable_truncation(max_length=MAX_LEN); tok.enable_padding(length=MAX_LEN, pad_token="[PAD]")
    return tok

def make_config():
    return BertConfig(vocab_size=12000, hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                      intermediate_size=1024, max_position_embeddings=MAX_LEN, num_labels=NUM_LABELS, pad_token_id=0)

def align_labels(text, entities, tokenizer):
    enc = tokenizer.encode(text); offsets = enc.offsets; tokens = enc.ids
    labels = [LABEL2ID["O"]]*len(offsets)
    for ent in entities:
        first = True
        for i,(s,e) in enumerate(offsets):
            if s == e == 0: continue
            if s >= ent["start"] and e <= ent["end"] and e > s:
                labels[i] = LABEL2ID[("B-" if first else "I-")+ent["label"]]; first = False
    for i,(s,e) in enumerate(offsets):
        if s == e == 0: labels[i] = -100
    return tokens, enc.attention_mask, labels

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer): self.records, self.tokenizer = records, tokenizer
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]; t,a,l = align_labels(r["text"], r["entities"], self.tokenizer)
        return {"input_ids":torch.tensor(t), "attention_mask":torch.tensor(a), "labels":torch.tensor(l)}

def extract_spans(label_ids):
    spans, cur_type, cur_start = [], None, None
    for i,lid in enumerate(label_ids):
        if lid == -100:
            if cur_type is not None: spans.append((cur_start,i-1,cur_type)); cur_type=None
            continue
        label = ID2LABEL[lid]
        if label == "O":
            if cur_type is not None: spans.append((cur_start,i-1,cur_type)); cur_type=None
        elif label.startswith("B-"):
            if cur_type is not None: spans.append((cur_start,i-1,cur_type))
            cur_type = label[2:]; cur_start = i
        else:
            et = label[2:]
            if cur_type != et:
                if cur_type is not None: spans.append((cur_start,i-1,cur_type))
                cur_type = et; cur_start = i
    if cur_type is not None: spans.append((cur_start,len(label_ids)-1,cur_type))
    return spans

def entity_level_prf(model, loader):
    model.eval(); tp=fp=fn=0; per_type={}
    with torch.no_grad():
        for b in loader:
            preds = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.argmax(-1)
            for i in range(preds.size(0)):
                g = set(extract_spans(b["labels"][i].tolist())); p = set(extract_spans(preds[i].tolist()))
                tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
                for s in g|p: per_type.setdefault(s[2],{"tp":0,"fp":0,"fn":0})
                for s in g&p: per_type[s[2]]["tp"]+=1
                for s in p-g: per_type[s[2]]["fp"]+=1
                for s in g-p: per_type[s[2]]["fn"]+=1
    pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
    return {"precision":pr,"recall":rc,"f1":f1,"tp":tp,"fp":fp,"fn":fn}, per_type

def load_model(path):
    m = BertForTokenClassification(make_config())
    m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); return m

"""Builds IA2_Report.md from the result files (every number in the report is read from results/*.json|csv)."""
import json, csv, math, hashlib, os
from scipy.stats import fisher_exact
R = "./results"
J = lambda n: json.load(open(f"{R}/{n}"))
base, base_es, adv = J("baseline_ia2.json"), J("baseline_ia2_es.json"), J("adv_training_ia2.json")
dp, ctl, red = J("dpsgd_results.json"), J("dp_controls.json"), J("redaction_eval.json")
ldp, cp, ka = J("ldp_results.json"), J("compressive_privacy.json"), J("kanonymity_summary.json")
lat = list(csv.DictReader(open(f"{R}/kanonymity_lattice.csv")))
f3 = lambda x: f"{x:.3f}"; pc = lambda x: f"{100*x:.1f}%"
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); a = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-a)/d, (c+a)/d)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
D = "./data"

# ---------- derived numbers ----------
flips = {k: round(v["all"]["flip_rate"]*v["all"]["n"]) for k, v in red.items()}
fe = fisher_exact([[flips["ia2_baseline_es"], red["ia2_baseline_es"]["all"]["n"]-flips["ia2_baseline_es"]],
                   [flips["ia2_adv_trained"], red["ia2_adv_trained"]["all"]["n"]-flips["ia2_adv_trained"]]])[1]
miss = {k: v["clean_miss_pii"][0]/v["clean_miss_pii"][1] for k, v in red.items()}
T = base["test"]; TE = base_es["test"]; TA = adv["test"]
def row_model(name, label, t):
    r = red[name]; a = r["all"]; lo, hi = wilson(flips[name], a["n"])
    return (f"| {label} | {f3(t['f1'])} | {f3(t['precision'])} | {f3(t['recall'])} | {a['n']} | {pc(a['flip_rate'])} ({pc(lo)} to {pc(hi)}) | "
            f"{pc(r['pii']['extra_leak_rate'])} | {f3(a['avg_conf_drop'])} |")

md = []
A = md.append
A('''<div class="hdr">
<h1>Security and Privacy in Machine Learning</h1>
<div class="ia">IA2</div>
<div class="sub">Privacy-Preserving Defences for Adversarially Robust Privacy-Sensitive NER on Wikipedia Biographies of Living Persons</div>
<div class="college">KJ Somaiya School of Engineering</div>
<div class="nm"><span><b>Name</b>: Ayaan Banatwalla</span></div>
<div class="nm"><span><b>Roll No</b>: 16014223025</span></div>
<div class="gh"><b>Github:</b> <a href="https://github.com/ayaanbanatwalla-lab/wikipedia-blp-privacy-ner-adversarial">https://github.com/ayaanbanatwalla-lab/wikipedia-blp-privacy-ner-adversarial</a></div>
</div>
''')

# ================= 1 =================
A('''## 1. Problem Refinement and Objectives

**Revised problem statement.** Usecase: *To what extent do Wikipedia Biographies of Living Persons (BLPs) violate data protection regulations (GDPR, CCPA)?* IA-1 showed that the automated privacy-NER auditor needed to answer this can be degraded by small, meaning-preserving context edits. IA-2 asks the next question: **which privacy-preserving techniques from the course can protect the people described in a BLP corpus and the auditor itself, and what do they cost in accuracy and robustness?** Concretely, the study now covers three layers: (i) the *auditor model* (NER trained with and without differential privacy, and adversarially), (ii) the *redaction pipeline* that uses the auditor's output to mask personal data, and (iii) the *released structured data* (birth date and birthplace of each BLP).

**Revised objectives.**

1. Re-establish a reproducible baseline on the finalised dataset (Section 3) and keep IA-1's attack and adversarial-training results comparable to it.
2. **Module 2.** Train the NER model with DP-SGD at several privacy budgets and measure the utility and robustness cost against the non-private model.
3. **Module 4.** Measure how much personal data a NER-driven redaction pipeline leaves exposed, with and without the attack, and measure re-identification risk of the released birth-date and birthplace attributes with k-anonymity and generalisation.
4. **Module 3.** Demonstrate local differential privacy (randomised response and the piecewise mechanism) on the same attributes, and compare it with central DP.
5. **Module 5.** Demonstrate compressive privacy: a projection that keeps a utility attribute while hiding a sensitive one.
6. Report evaluation, limitations and reproducibility instructions in one technical report.

**Research questions.**

- **RQ1.** What accuracy does the NER auditor lose when trained with DP-SGD at each privacy budget, and does DP change its robustness to the attack?
- **RQ2.** How much personal data does NER-driven redaction leave exposed on clean text, and how much extra leaks under attack, for the baseline, adversarially trained and DP models?
- **RQ3.** How identifiable are BLPs from (birth date, birthplace), and what generalisation makes the release k-anonymous at what information loss?
- **RQ4.** How does estimation error under local DP compare with central DP on this data, and how well can a compressive-privacy projection separate utility from leakage?

**Technical requirements.** Python 3.13, PyTorch (CPU), Hugging Face Transformers and Tokenizers, Opacus (DP-SGD with RDP accounting), spaCy `en_core_web_sm`, WordNet 3.0, scikit-learn, NumPy/SciPy, Matplotlib. Everything runs on two CPU cores in under three hours end to end. All random seeds are fixed (42).

### Justification of changes from IA-1

| Item | IA-1 | IA-2 | Why it changed |
| --- | --- | --- | --- |
| Scope | Attack and defence on the NER model (Module 1 core) | Adds privacy-preserving training, redaction, anonymisation, LDP and compressive privacy | IA-1 mapped Modules 2 to 5 analytically and planned experiments for IA-2 |
| Baseline | Test F1 0.216 (P 0.163, R 0.323) | Test F1 %s (P %s, R %s) | Rebuilding the pipeline from the raw file after an environment reset exposed that the IA-1 run used a date-aware infobox matcher not present in the notebook cell. The matcher is now in the code (`common.py`) and the baseline is re-trained with the identical architecture and hyper-parameters. Numbers in IA-1 are kept as history and not mixed with IA-2 numbers |
| Defence comparison | Adversarial model (best-validation checkpoint) compared with a final-epoch baseline | Adversarial model compared with an **early-stopped baseline** as well | A final-epoch baseline is over-trained and makes early stopping look like an adversarial-training benefit |
| Redaction metric | Same-type entity flip | Adds a type-agnostic redaction check: is the target still covered by *any* predicted entity | A redactor masks any detected span, so type changes do not leak data |
| Module depth | M2 to M5 analytical | M2 and M4 full experiments; M3 and M5 controlled demonstrations | Agreed scope for the IA-2 deadline; the demonstrations use the same 299-record quasi-identifier table so they stay tied to the use case |
| Manuscript | Not required | The IEEE paper is updated as the manuscript; no separate IP draft | Agreed scope |
''' % (f3(T['f1']), f3(T['precision']), f3(T['recall'])))

# ================= 2 =================
A('''## 2. Integration of Course Concepts

### Final module-wise mapping

| Module | Technique used | Where it acts on the use case | Depth | Evidence |
| --- | --- | --- | --- | --- |
| 1. Privacy Consideration in ML | Black-box evasion attack, adversarial training, GDPR/CCPA risk scoring, privacy-oriented data mining (taxonomy) | The auditor is the ML system under attack; risk scoring turns misses into compliance impact | Implemented (IA-1), re-run on the final dataset | `08_adv_train.py`, `09_redaction_eval.py`, `results/adv_training_ia2.json` |
| 2. Differential Privacy | DP-SGD with per-sample clipping, Gaussian noise, RDP accounting (Opacus), epsilon sweep with ablation controls | Bounds what the NER model can reveal about any one biography | **Implemented and evaluated** | `dp.py`, `04_run_dp.py`, `04b_dp_controls.py`, `results/dpsgd_results.json` |
| 3. Local Differential Privacy | k-ary randomised response; piecewise mechanism (with Laplace as comparison) | Each BLP record is perturbed before any aggregation of birth country and birth year | **Controlled demonstration** | `06_ldp_demo.py`, `results/ldp_results.json` |
| 4. Privacy-Preserving Data Mining | NER-driven redaction (modifying the input), k-anonymity with generalisation hierarchies and suppression (publishing data) | Redaction of text; anonymised release of (birth date, birthplace) | **Implemented and evaluated** | `09_redaction_eval.py`, `05_kanonymity.py`, `results/redaction_eval.json`, `results/kanonymity_lattice.csv` |
| 5. Compressive Privacy | PCA, DCA and DUCA projections of paragraph features | Keeps birth continent (utility) while hiding birth decade (sensitive) | **Controlled demonstration** | `07_compressive_privacy.py`, `results/compressive_privacy.json` |

### Architecture and workflow

![IA-2 workflow](figures/fig_workflow.png)

*Figure 1. Data flows from the live scraper to the NER model; Modules 2 and 4 attach to the model, the attack and the data; the Module 3 and 5 demonstrations use the same 299-record quasi-identifier table.*

### How the concepts are integrated

- **Module 1 to Module 2.** Adversarial training defends against *evasion*, a test-time attack. DP-SGD defends against *training-data leakage*. They answer different threats, so IA-2 evaluates DP models under the same evasion attack to see whether one defence affects the other.
- **Module 1 to Module 4.** The auditor's output is used to redact text. An evasion that stops a detection therefore becomes a *leak* in the redacted output. This links an ML-security result (attack success) to a privacy-preserving-data-mining result (residual exposure).
- **Module 4 to Modules 3 and 5.** After redaction, structured attributes such as birth date and birthplace remain. k-anonymity (Module 4), local DP (Module 3) and compressive privacy (Module 5) are three different ways to limit what those attributes reveal, so they are compared on the **same 299 records**.
- **Why Modules 3 and 5 are demonstrations, not full experiments.** Local DP needs many data holders to randomise their own values; compressive privacy needs a utility task and a privacy task with enough labelled data. With 299 records both are shown as controlled demonstrations with measured error, rather than as claims about production-scale performance. Section 4 states what each demonstration does and does not show.
''')

# ================= 3 =================
tr, va, te = (sum(1 for _ in open(f"{D}/ner_{s}.jsonl")) for s in ("train", "val", "test"))
A(f'''## 3. Dataset Finalisation and Individual Implementation

### Final dataset and provenance record

The final dataset is the same **500 live-scraped English Wikipedia BLP intro paragraphs** as in IA-1, with the annotation refined as described below. No ready-made dataset is used.

| Attribute | Record |
| --- | --- |
| Name and version | `raw_blp_data.jsonl`, version 1 (single scrape, 500 retained of 641 candidates processed) |
| SHA-256 | `{sha(f"{D}/raw_blp_data.jsonl")}` |
| Size | 524,423 bytes, 500 records, about 80 words per paragraph |
| Collection | Live scrape through the MediaWiki API and Wikidata API; file timestamp 7 October 2026 |
| Source and licence | English Wikipedia text under CC BY-SA 4.0 and Wikidata under CC0 (Wikimedia terms); non-commercial coursework use; every record keeps its source URL |
| Access restrictions | None beyond the licences. Only public Wikipedia content is used, only intro paragraphs and public infobox fields, and no private data was collected. Results are about a model and a method, not findings about any individual article |
| Derived files | `ner_train.jsonl` (350 records, SHA-256 `{sha(f"{D}/ner_train.jsonl")[:16]}...`), `ner_val.jsonl` (75, `{sha(f"{D}/ner_val.jsonl")[:16]}...`), `ner_test.jsonl` (75, `{sha(f"{D}/ner_test.jsonl")[:16]}...`), `privbert_tokenizer-vocab.txt` (12,000 WordPiece tokens, `{sha(f"{D}/privbert_tokenizer-vocab.txt")[:16]}...`) |
| Sample used | All 500 records (70/15/15 article-level split, seed 42). One paragraph per person, so a record is also one data subject |
| Annotation status | Silver-standard (regex, spaCy `en_core_web_sm`, infobox cross-reference), `human_verified: false` |

**Refinement made in IA-2.** Infobox birth dates are ISO strings (for example `1964-02-22`) while article text writes `22 February 1964`. The matcher now converts each ISO date to the textual forms before cross-referencing. This yields **{ka['n']} birth-date plus birthplace records** for Modules 3 to 5 and 325 `DATE_OF_BIRTH` spans in the annotation (train 234, validation 43, test 48).

**Data-quality issues found and handled.** Some infobox fields contain wiki markup (for example `U.S.{{cite web`, external-link fragments, bracketed notes); `qi.py` cleans these before generalisation. Some infobox values are unusable (for example a spouse field holding a database title) and are not used as labels.

### Preprocessing scripts and final code

| File | Purpose |
| --- | --- |
| `scrape_wikipedia_blps.py` | Live scraper (IA-1): titles from Category:Living_people, extract and wikitext per page, Wikidata living-status check |
| `common.py` | Cleaning, silver annotation (including the date matcher), 70/15/15 split, tokenizer training, BIO alignment, model and metric code |
| `01_prepare_data.py` | Builds `ner_*.jsonl` and the tokenizer from `raw_blp_data.jsonl` |
| `02_train_baseline.py`, `02b_train_baseline_es.py` | Baseline (final epoch) and early-stopped baseline |
| `attack.py` | Black-box context-only attack and redaction check |
| `08_adv_train.py` | Adversarial training |
| `dp.py`, `04_run_dp.py`, `04b_dp_controls.py` | DP-SGD (Module 2) and controls |
| `05_kanonymity.py`, `qi.py` | k-anonymity lattice (Module 4) |
| `09_redaction_eval.py` | NER-driven redaction leakage (Module 4) |
| `06_ldp_demo.py` | Local DP demonstration (Module 3) |
| `07_compressive_privacy.py` | Compressive-privacy demonstration (Module 5) |
| `10_workflow_fig.py`, `build_ia2.py` | Workflow figure; builds this report's tables from the result files |

### Execution instructions

```
cd ia2                                         # all paths in this report are relative to the ia2/ folder of the repository
pip install torch transformers tokenizers spacy opacus scikit-learn scipy matplotlib nltk wn==0.0.23
python -m spacy download en_core_web_sm        # WordNet 3.0 must be available to nltk
python 01_prepare_data.py                      # splits (about 1 min); reuses the shipped tokenizer vocabulary
python 02_train_baseline.py                    # about 6 min on 2 CPU cores
python 02b_train_baseline_es.py
python 08_adv_train.py                         # about 20 min
python 04_run_dp.py 16,4,64 15 3e-3 64         # about 8 min per epsilon
python 04_run_dp.py 16 40 3e-3 64 _e40         # longer schedule, about 20 min
python 04b_dp_controls.py
python 09_redaction_eval.py ia2_baseline ia2_baseline_es ia2_adv_trained ia2_dp_eps16 ia2_dp_eps64
python 05_kanonymity.py ; python 06_ldp_demo.py ; python 07_compressive_privacy.py
python build_ia2.py
```

All outputs are written to `results/` and `figures/`. Seeds are fixed to 42; exact floating-point results can differ slightly across hardware and library versions. WordPiece tokenizer training is not bit-reproducible, so the trained vocabulary file `data/privbert_tokenizer-vocab.txt` is shipped and reused (`01_prepare_data.py` only retrains it if the file is missing); the checkpoints depend on that exact file.

### Individual contribution statement

This work was done individually by me: problem formulation (done with the help of my teacher), the live scraper, annotation pipeline and the date-matcher refinement, model training, the DP-SGD, adversarial-training, redaction, k-anonymity, local-DP and compressive-privacy experiments, the analysis, this report and the research paper.
''')

# ================= 4 =================
A(f'''## 4. Evaluation, Comparison and Critical Analysis

**Metrics.** NER: entity-level precision, recall and F1 with exact span and type match, on the 75-paragraph test split. Attack: of the entities the model detects correctly, the share whose same-type detection is removed (flip rate, with 95% Wilson interval), the mean drop in model confidence, and the **extra leak rate**, the share of correctly redacted privacy-relevant entities (PERSON plus the 15 privacy categories) that no longer have any predicted entity covering them after the attack. Redaction: **clean miss rate**, the share of gold privacy-relevant entities not covered by any predicted entity on clean text. DP: epsilon at fixed delta. k-anonymity: share of records in classes smaller than k and normalised generalisation height as information loss. LDP: L1 error of the country histogram and mean absolute error of the mean birth year, averaged over 500 randomisations. Compressive privacy: 5-fold, 3-seed cross-validated accuracy of a utility classifier and of a privacy adversary.

### 4.1 Baseline, adversarial training and attack (Module 1)

The attack is the IA-1 context-only black-box attack (at most 3 synonym substitutions, at most 60 queries, entity tokens never edited), re-run against each model on the test split.

| Model | Test F1 | P | R | Attackable entities | Flip rate (95% CI) | Extra leak, privacy entities | Mean confidence drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
{row_model("ia2_baseline", "Baseline (final epoch 25)", T)}
{row_model("ia2_baseline_es", f"Baseline, early stopped (epoch {base_es['best_epoch']})", TE)}
{row_model("ia2_adv_trained", f"Adversarially trained (epoch {adv['best_epoch']}, {adv['n_adv']} adversarial paragraphs added)", TA)}

*Early-stopped baseline versus adversarially trained: flip rates {pc(red['ia2_baseline_es']['all']['flip_rate'])} versus {pc(red['ia2_adv_trained']['all']['flip_rate'])}, Fisher exact p = {fe:.2f}.*

**Reading.** Adversarial training raises test F1 from {f3(TE['f1'])} (early-stopped baseline) to {f3(TA['f1'])} and the flip rate is lower, but the difference is **not statistically significant** ({flips['ia2_baseline_es']} versus {flips['ia2_adv_trained']} flips). The mean confidence drop is actually *larger* for the adversarially trained model, which means its confidence is more fragile even where the final label survives. The honest conclusion is the same as in IA-1: full flips are rare for context-only edits with this model, and the evidence for an adversarial-training benefit is weak at this sample size.

### 4.2 Differential privacy: DP-SGD (Module 2)

Setting: per-sample gradient clipping norm C = 1.0, Gaussian noise, Poisson sampling at rate q = 1/6 (expected batch about 58), 15 epochs (90 steps) unless stated, AdamW with learning rate 3e-3, delta = 1e-3 (below 1/n = 1/350), RDP accountant. Because there is one paragraph per person, **example-level DP is person-level DP** here.

| Model | Epsilon | Noise multiplier | Test F1 | Test P | Test R | Attackable entities | Flip rate | Mean confidence drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Non-private baseline, early stopped (batch 16, lr 5e-4) | infinite | none | {f3(TE['f1'])} | {f3(TE['precision'])} | {f3(TE['recall'])} | {red['ia2_baseline_es']['all']['n']} | {pc(red['ia2_baseline_es']['all']['flip_rate'])} | {f3(red['ia2_baseline_es']['all']['avg_conf_drop'])} |
| Same optimiser, no clipping, no noise | infinite | none | {f3(ctl['no_clip_no_noise']['test']['f1'])} | {f3(ctl['no_clip_no_noise']['test']['precision'])} | {f3(ctl['no_clip_no_noise']['test']['recall'])} | n/a | n/a | n/a |
| Same optimiser, clipping only | infinite | 0 | {f3(ctl['clip_only']['test']['f1'])} | {f3(ctl['clip_only']['test']['precision'])} | {f3(ctl['clip_only']['test']['recall'])} | n/a | n/a | n/a |
| DP-SGD | 64 | {dp['64']['sigma']:.2f} | {f3(dp['64']['test']['f1'])} | {f3(dp['64']['test']['precision'])} | {f3(dp['64']['test']['recall'])} | {dp['64']['attack']['n']} | {pc(dp['64']['attack']['flip_rate'])} | {f3(dp['64']['attack']['avg_conf_drop'])} |
| DP-SGD | 16 | {dp['16']['sigma']:.2f} | {f3(dp['16']['test']['f1'])} | {f3(dp['16']['test']['precision'])} | {f3(dp['16']['test']['recall'])} | {dp['16']['attack']['n']} | {pc(dp['16']['attack']['flip_rate'])} | {f3(dp['16']['attack']['avg_conf_drop'])} |
| DP-SGD, 40 epochs (240 steps) | 16 | {dp['16_e40']['sigma']:.2f} | {f3(dp['16_e40']['test']['f1'])} | {f3(dp['16_e40']['test']['precision'])} | {f3(dp['16_e40']['test']['recall'])} | {dp['16_e40']['attack']['n']} | {pc(dp['16_e40']['attack']['flip_rate'])} | {f3(dp['16_e40']['attack']['avg_conf_drop'])} |
| DP-SGD | 4 | {dp['4']['sigma']:.2f} | {f3(dp['4']['test']['f1'])} | {f3(dp['4']['test']['precision'])} | {f3(dp['4']['test']['recall'])} | {dp['4']['attack']['n']} | {pc(dp['4']['attack']['flip_rate'])} | {f3(dp['4']['attack']['avg_conf_drop'])} |

![DP-SGD sweep](figures/fig_dp.png)

*Figure 2. Test F1 against privacy budget. The dashed lines are the non-private references.*

**Reading.** DP-SGD costs a lot of accuracy on this task: the best private model (epsilon 64, F1 {f3(dp['64']['test']['f1'])}; epsilon 16, F1 {f3(dp['16']['test']['f1'])}) is well under half of the non-private {f3(TE['f1'])}, and at epsilon 4 it learns almost nothing. Four observations qualify that number.

1. **The controls fail too.** With the same optimiser, batch size and 90-step schedule, the model trained *without* clipping and noise, and the one trained with clipping only, both reach F1 0.000: they are still predicting only the majority label `O` after 90 steps. The noisy runs escape that plateau. So part of the gap to the non-private baseline comes from the short schedule and the optimiser change that DP training needs, and the gap cannot be read as the pure cost of privacy.
2. **More steps did not help.** The 40-epoch run at epsilon 16 gives F1 {f3(dp['16_e40']['test']['f1'])}, lower than the 15-epoch run, because for a fixed epsilon the noise multiplier must grow with the number of steps ({dp['16']['sigma']:.2f} to {dp['16_e40']['sigma']:.2f}). At epsilon 16 the model reaches a ceiling of roughly F1 0.1.
3. **Why the ceiling is low.** There are only 350 training paragraphs, a 39-label BIO task dominated by `O`, and a model trained from scratch with no pretrained knowledge, so per-step noise swamps the signal.
4. **The attack comparison is not reliable.** DP models correctly detect only {dp['4']['attack']['n']} to {dp['64']['attack']['n']} entities, so little remains to attack and no flips occur. DP-SGD should be read here as a privacy-of-training result, not a robustness result.

### 4.3 NER-driven redaction and k-anonymity (Module 4)

**Redaction leakage.** The redactor masks every span the NER model predicts. Privacy-relevant gold entities are PERSON plus the 15 privacy categories (317 instances in the test split).

| Model | Clean miss rate (privacy entities not covered) | Extra leak under attack (correctly redacted, then uncovered) | Mean confidence drop on privacy entities |
| --- | --- | --- | --- |
''' + "\n".join(
    f"| {lab} | {pc(miss[k])} ({red[k]['clean_miss_pii'][0]} of {red[k]['clean_miss_pii'][1]}) | {pc(red[k]['pii']['extra_leak_rate'])} (n={red[k]['pii']['n']}) | {f3(red[k]['pii']['avg_conf_drop'])} |"
    for k, lab in (("ia2_baseline", "Baseline (final epoch)"), ("ia2_baseline_es", "Baseline, early stopped"),
                   ("ia2_adv_trained", "Adversarially trained"), ("ia2_dp_eps64", "DP-SGD, epsilon 64"), ("ia2_dp_eps16", "DP-SGD, epsilon 16"))) + f'''

**Reading.** The dominant leak is **not the attack**: even on clean text, the best model leaves {pc(min(miss.values()))} of privacy-relevant entities uncovered, and DP models leave {pc(min(miss['ia2_dp_eps16'], miss['ia2_dp_eps64']))} to {pc(max(miss['ia2_dp_eps16'], miss['ia2_dp_eps64']))}. The attack adds a small further leak (0 to {pc(max(v['pii']['extra_leak_rate'] for v in red.values()))} of the entities that were redacted). For this pilot, an under-trained auditor misses far more than an attacker can make it miss, so the first priority for a real GDPR/CCPA audit is recall, and robustness comes second.

**k-anonymity of the released attributes.** The quasi-identifiers are birth date and birthplace of the {ka['n']} BLPs that have both. Generalisation hierarchies: date (exact, month, year, 5-year band, decade, suppressed) and place (full, region and country, country, continent, suppressed). The lattice has 6 x 5 = 30 nodes; information loss is the normalised sum of generalisation levels.

| Release | Records unique | Smallest class size reached with at most 5% suppression | Information loss |
| --- | --- | --- | --- |
| Exact date and full place | {pc(ka['raw_unique_pct'])} | 1 (no anonymity) | 0 |
| Year and place suppressed | {pc(ka['best']['2']['pct_unique'])} | k = 2 | {ka['best']['2']['info_loss']:.2f} |
| 5-year band and place suppressed | {pc(ka['best']['3']['pct_unique'])} | k = 3 and k = 5 (suppress {pc(ka['best']['5']['suppress_for_k5'])} for k = 5) | {ka['best']['5']['info_loss']:.2f} |
| Date suppressed, continent kept | 0.0% | k = 10 (suppress {pc(ka['best']['10']['suppress_for_k10'])}) | {ka['best']['10']['info_loss']:.2f} |

![k-anonymity trade-off](figures/fig_kanon.png)

*Figure 3. Each point is one of the 30 generalisation nodes. Higher information loss buys a larger share of records in classes of size at least k.*

**Reading.** **All {ka['n']} (birth date, birthplace) pairs are unique**, so a BLP is fully re-identifiable from them. With 299 records, meaningful k-anonymity needs heavy generalisation: the place must be dropped or reduced to a continent and the date reduced to a year or coarser. This is the classic small-sample limitation of k-anonymity and one reason the report also tests local DP and compressive privacy. k-anonymity also gives no protection against attribute linkage with other public sources, which is a limitation of the method.

### 4.4 Local differential privacy (Module 3, controlled demonstration)

Each record's birth country (top 8 countries plus Other) is perturbed with k-ary randomised response, and its birth year with the piecewise mechanism (with the Laplace mechanism as comparison), before aggregation. Central DP (Laplace on the aggregate) is the reference. Errors are averaged over 500 runs.

| Epsilon | Histogram L1 error, LDP randomised response (n = 299) | Histogram L1 error, central Laplace (n = 299) | Mean birth year error in years, piecewise (n = 299) | Laplace LDP (n = 299) | Piecewise (n = 10,000, bootstrap) |
| --- | --- | --- | --- | --- | --- |
''' + "\n".join(
    f"| {e} | {f3(ldp['real_n299']['grr_L1'][i])} | {f3(ldp['real_n299']['central_L1'][i])} | {ldp['real_n299']['pm_mae_years'][i]:.2f} | {ldp['real_n299']['lap_mae_years'][i]:.2f} | {ldp['bootstrap_n10000']['pm_mae_years'][i]:.2f} |"
    for i, e in enumerate(ldp['real_n299']['eps'])) + f'''

![LDP demonstration](figures/fig_ldp.png)

*Figure 4. Error against epsilon for categorical and numeric attributes, on the real 299 records and on a 10,000-record bootstrap.*

**Reading.** At epsilon 0.5 the LDP histogram error is {f3(ldp['real_n299']['grr_L1'][0])} (the maximum possible is 2), so the estimate is essentially noise; it falls below 0.1 only at epsilon 4. The piecewise mechanism beats Laplace for the numeric attribute at every epsilon, as expected from the theory. Local DP costs far more accuracy than central DP at the same epsilon because noise is added to every record rather than once to the aggregate, and the error shrinks roughly with the square root of n (10,000 records give about 6 times lower error). Local DP is only useful with a large user population, which a 500-page audit does not have. The 10,000-record rows are a bootstrap of the real values and show scaling, not new data.

### 4.5 Compressive privacy (Module 5, controlled demonstration)

Features are TF-IDF vectors of the paragraph text (years generalised to decade tokens), reduced to 100 dimensions. **Utility:** birth continent (4 classes, majority baseline {pc(cp['chance_U'])}). **Privacy leak:** birth-decade group (4 classes, majority baseline {pc(cp['chance_P'])}). PCA is unsupervised; DCA keeps directions that separate the utility classes; DUCA also penalises directions that separate the privacy classes.

| Representation | m | Utility accuracy | Privacy adversary accuracy |
| --- | --- | --- | --- |
| Uncompressed (100 dims) | 100 | {pc(cp['RAW'][0])} | {pc(cp['RAW'][1])} |
''' + "\n".join(
    f"| {meth} | {cp['m'][i]} | {pc(cp[meth][i][0])} | {pc(cp[meth][i][1])} |"
    for meth in ("PCA", "DCA", "DUCA") for i in (0, 2, 4, 7)) + f'''

![Compressive privacy](figures/fig_cp.png)

*Figure 5. Left: utility accuracy; right: accuracy of an adversary predicting birth decade, against projected dimensions m.*

**Reading.** The uncompressed features leak the birth decade ({pc(cp['RAW'][1])} against a {pc(cp['chance_P'])} majority baseline) and support the utility task at {pc(cp['RAW'][0])}. DUCA with m = 3 reaches utility {pc(cp['DUCA'][2][0])} while pushing the adversary to {pc(cp['DUCA'][2][1])}, which is about the majority baseline, so on this data the projection keeps the utility signal and removes the decade signal. PCA at small m keeps neither signal. The result is from 299 records with cross-validation, so the gaps of a few points are within noise; the large gaps (leakage {pc(cp['RAW'][1])} to about {pc(cp['DUCA'][2][1])}) are the finding. It is a demonstration that the method works as a utility/privacy trade-off tool, not a claim about general text.

### 4.6 Privacy-utility summary and critical analysis

| Technique | Privacy gain | Utility cost | Main caveat |
| --- | --- | --- | --- |
| Adversarial training (M1) | Lower flip rate, not significant | F1 up with early stopping | Confidence is more fragile; small sample |
| DP-SGD (M2) | Formal epsilon bound for each BLP | F1 {f3(dp['16']['test']['f1'])} at epsilon 16 against {f3(TE['f1'])} non-private | Tiny data; short schedule confound; attack metrics unreliable |
| k-anonymity (M4) | 100% unique down to k of 2 to 10 | Place dropped, date coarsened | Small-sample; linkage attacks remain |
| LDP (M3) | No trusted collector needed | Error of 0.8 L1 at epsilon 1 | Needs a large population |
| Compressive privacy (M5) | Decade leakage {pc(cp['RAW'][1])} to about {pc(cp['DUCA'][2][1])} | Utility {pc(cp['RAW'][0])} to {pc(cp['DUCA'][2][0])} (no loss) | Small data; one utility and one privacy task |

**Limitations.** (1) Silver-standard labels: all F1 numbers measure agreement with weak labels, not truth. (2) 75 test paragraphs: confidence intervals are wide and the 15 privacy categories have near-zero support apart from `DATE_OF_BIRTH` (48), `EMPLOYMENT` (12), `NATIONALITY` (4) and `FINANCIAL` (2). (3) The DP runs use different optimiser settings from the non-private baseline (larger batches, higher learning rate, fewer steps) because DP needs them to learn at all; the matched non-private controls fail under those settings, so the privacy cost cannot be isolated exactly. (4) Single seed per configuration. (5) The attack is context-only and limited to 3 substitutions, so low flip rates do not prove robustness to stronger attacks. (6) The Module 3 and 5 results are demonstrations on one small table.

**Improvements.** Human-verified labels for the privacy categories; a larger scrape (the scraper already supports 5,000 to 10,000 pages); a pretrained encoder fine-tuned with DP-SGD (the main fix for the utility loss); repeated seeds with confidence intervals; stronger attacks including entity-adjacent edits; and releasing structured attributes with differentially private aggregate statistics rather than record-level k-anonymity.
''')

# ================= 5 =================
A(f'''## 5. Final Documentation and Research Outcome

### Conclusions

1. **Answer to the use case at this scale.** An automated privacy auditor built the way a real audit would be built (a NER model over live Wikipedia text) is limited more by recall than by attack: on clean text it leaves at least {pc(min(miss.values()))} of privacy-relevant entities uncovered, and context-only attacks add a few percentage points of extra leakage.
2. **Module 2.** DP-SGD gives a formal per-person bound but at this data scale it costs most of the model's accuracy (F1 {f3(dp['16']['test']['f1'])} at epsilon 16 versus {f3(TE['f1'])}), so privacy-preserving training is only practical here with a pretrained encoder or much more data. The accuracy gap cannot be fully attributed to privacy, because the matched non-private controls with the same short schedule also failed (F1 0.000).
3. **Module 4.** Every BLP is unique under (birth date, birthplace); k-anonymity of 5 or more needs the date coarsened to a 5-year band and the place removed.
4. **Module 3.** Local DP needs thousands of records to be useful; at 299 records its error is large.
5. **Module 5.** A discriminant projection can keep a utility attribute and remove a sensitive one: leakage fell from {pc(cp['RAW'][1])} to about {pc(cp['DUCA'][2][1])} with no utility loss on this data.

### Reproducibility

Raw data hash, derived-file hashes, seeds, library versions and run order are recorded in Section 3 and in the repository README. Every table in this report is generated by `build_ia2.py` from the files in `results/`, so no number is typed by hand. Pretrained weights are not used; the model and tokenizer are trained from scratch. Three representative checkpoints (`ia2_baseline_es`, `ia2_adv_trained`, `ia2_dp_eps16`) are in `models/` as zip files; the other checkpoints are regenerated by the scripts.

### Research outcome

The updated IEEE paper (Section "Privacy-Preserving Extensions") is the manuscript for this work; no separate IP draft was prepared.

### References

1. Regulation (EU) 2016/679 (GDPR), 2016.
2. California Consumer Privacy Act of 2018 and CPRA amendments (CCPA), 2020.
3. M. Abadi et al., "Deep learning with differential privacy," ACM CCS, 2016.
4. A. Yousefpour et al., "Opacus: user-friendly differential privacy library in PyTorch," 2021.
5. L. Sweeney, "k-anonymity: a model for protecting privacy," Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems, 2002.
6. T. Wang et al., "Collecting and analyzing multidimensional data with local differential privacy," IEEE ICDE, 2019.
7. S. Y. Kung, "Compressive privacy: from information/estimation theory to machine learning," IEEE Signal Processing Magazine, 2017.
8. D. Jin et al., "Is BERT really robust? A strong baseline for natural language attack on text classification and entailment," AAAI, 2020.
9. L. Li et al., "BERT-Attack: adversarial attack against BERT using BERT," EMNLP, 2020.
10. R. Shokri et al., "Membership inference attacks against machine learning models," IEEE S&P, 2017.
11. Wikipedia, "Biographies of living persons" policy and Category:Living_people.

## Evidence Index

| Rubric stage | Evidence | Path in repository |
| --- | --- | --- |
| 1. Problem refinement and objectives | Revised statement, objectives, research questions, requirements, change justification | Section 1 |
| 2. Integration of course concepts | Module mapping table, workflow diagram, evidence per module | Section 2; `figures/fig_workflow.png`; `results/dpsgd_results.json`, `results/redaction_eval.json`, `results/kanonymity_lattice.csv`, `results/ldp_results.json`, `results/compressive_privacy.json` |
| 3. Dataset finalisation and implementation | Provenance record, hashes, preprocessing and final code, execution instructions, contribution statement | Section 3; `raw_blp_data.jsonl`; `common.py`, `01_prepare_data.py` and the scripts listed in Section 3 |
| 4. Evaluation, comparison and critical analysis | Metrics, baseline and adversarial comparison, DP sweep and controls, redaction, k-anonymity, LDP, compressive privacy, limitations | Section 4; `results/*.json`, `results/*.csv`; `figures/fig_dp.png`, `fig_kanon.png`, `fig_ldp.png`, `fig_cp.png` |
| 5. Final documentation and research | Conclusions, reproducibility, references, updated manuscript | Section 5; `paper/research_paper.pdf` |
''')
open("./IA2_Report.md", "w", encoding="utf-8").write("\n".join(md))
print("written", sum(len(m) for m in md))

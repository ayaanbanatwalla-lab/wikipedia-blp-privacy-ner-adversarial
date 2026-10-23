# IA-2: Privacy-preserving defences for the Wikipedia BLP privacy-NER auditor

Security and Privacy in Machine Learning, KJ Somaiya School of Engineering. This folder holds the code, data, results and report for IA-2. The IA-1 material is in the repository root.

## Contents

| Path | What it is |
| --- | --- |
| `IA2_Report.pdf`, `IA2_Report.md` | Technical report (all tables are generated from `results/` by `build_ia2.py`) |
| `common.py`, `qi.py`, `attack.py`, `dp.py` | Shared pipeline, quasi-identifier hierarchies, black-box attack, DP-SGD trainer |
| `01_prepare_data.py` to `11_fig_dp.py` | Experiments, one per step (see the report, Section 3, for run order) |
| `data/` | `raw_blp_data.jsonl` (500 live-scraped BLP paragraphs), `ner_train/val/test.jsonl`, tokenizer vocabulary, taxonomy |
| `results/` | All metrics as JSON and CSV, and run logs |
| `figures/` | Figures used in the report |
| `models/` | Three checkpoints as zip files (`ia2_baseline_es`, `ia2_adv_trained`, `ia2_dp_eps16`); unzip into `data/` |
| `paper/` | Updated IEEE manuscript (source and PDF) |

## Run

```
cd ia2
pip install torch transformers tokenizers spacy opacus scikit-learn scipy matplotlib nltk wn==0.0.23
python -m spacy download en_core_web_sm
python 01_prepare_data.py && python 02_train_baseline.py && python 02b_train_baseline_es.py
python 08_adv_train.py
python 04_run_dp.py 16,4,64 15 3e-3 64 && python 04_run_dp.py 16 40 3e-3 64 _e40 && python 04b_dp_controls.py
python 09_redaction_eval.py ia2_baseline ia2_baseline_es ia2_adv_trained ia2_dp_eps16 ia2_dp_eps64
python 05_kanonymity.py && python 06_ldp_demo.py && python 07_compressive_privacy.py
python build_ia2.py
```

Dataset: SHA-256 of `data/raw_blp_data.jsonl` is `08303d7eca00bdf52defd7dec2813a9abb3e7af37734a64fddfc0f3269a5bc56`. Wikipedia text is CC BY-SA 4.0 and Wikidata is CC0. The tokenizer vocabulary is shipped because WordPiece training is not bit-reproducible; the checkpoints depend on it.

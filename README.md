# Adversarial Robustness of Privacy-Sensitive NER on Wikipedia BLPs

Security and Privacy in Machine Learning, Module 1: Privacy Consideration in Machine Learning
KJ Somaiya School of Engineering. Ayaan Banatwalla (16014223025)

Research question: *To what extent do Wikipedia Biographies of Living Persons (BLPs) violate data protection regulations (GDPR, CCPA)?* This project studies whether the privacy-sensitive NER model an audit would rely on can be fooled by context-only text changes, and what that does to privacy-risk scores.

## Repository layout

| Path | Contents |
| --- | --- |
| `scrape_wikipedia_blps.py` | Live scraper (MediaWiki API + Wikidata) |
| `raw_blp_data.jsonl` | 500 scraped BLP records |
| `privacy_ner_adversarial_research.ipynb` | Full pipeline: annotation, baseline NER, attack, risk scoring, adversarial training, statistics, figures |
| `models/` | Baseline and adversarially trained NER checkpoints (PyTorch `.pth`, zipped to fit GitHub's 25 MB upload limit; unzip before loading) |
| `results/` | Metrics (CSV/JSON), privacy taxonomy, risk scores, statistical tests |
| `figures/` | Figures 1 to 9 used in the paper |
| `demo/` | Before/after attack examples (3 real test-set paragraphs) |

## How to run

1. `python scrape_wikipedia_blps.py --num-pages 500 --out raw_blp_data.jsonl` (needs internet access to Wikipedia and Wikidata), or use the included `raw_blp_data.jsonl`.
2. Run `privacy_ner_adversarial_research.ipynb` top to bottom (seed 42). Outputs are written to the working directory.

## Headline results (test split)

- Baseline NER F1 0.216; on the attacked subset F1 falls 0.234 to 0.079 (66.1% relative drop).
- Full-flip attack success 3.7% (8 of 215 entities), McNemar p = 0.0078; DATE_OF_BIRTH 0 of 24 flipped.
- Adversarial training (best checkpoint): adversarial F1 +0.018, clean F1 -0.015.

## Notes

- Annotations are silver-standard (regex, spaCy, infobox cross-reference); `human_verified` is false.
- Wikipedia text is CC BY-SA; Wikidata is CC0. Data is used for coursework research only.
- GDPR/CCPA mapping is a research risk model, not legal advice.

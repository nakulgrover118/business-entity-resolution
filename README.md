# Business Entity Resolution — ML Challenge 2026

**Team Gym Bros** — Nakul Grover, Manjot Singh, Arshpreet Gill
**Final leaderboard score: 0.990336** (macro F0.5 per Source-1 entity, US / India / France)

Given business records (name + address) from three noisy sources with no shared identifiers, find every
Source-2 / Source-3 record that refers to the same business as each entity of the deduplicated Source 1.
The test set adds a country (France) that never appears in training.

## Approach

```
normalise ─► two-pass TF-IDF blocking ─► learned candidate filter ─► LightGBM matcher (3-seed bag)
          ─► 5 multilingual cross-encoders on the hard pairs ─► LightGBM blend
          ─► expected-F0.5 decision per entity ─► team ensemble + label-validated French rules
```

- **Normalisation learned from the data:** an Indic-script → Latin dictionary and a state
  canonicalisation learned from the training pairs (unsupervised for France); legal forms, aliases,
  web names and typos are handled explicitly.
- **Blocking:** sparse IDF-weighted search over names, name bigrams, addresses and name × city /
  house-number keys, plus an address-only pass. A second, much wider search covers the records the
  first pipeline leaves unmatched. A learned cheap-feature filter then keeps **4.4 candidates per entity**
  (train pair recall 99.07 % US / 98.65 % India before the filter).
- **Matcher:** two-stage LightGBM with "competition" features (each record belongs to at most one
  entity) and word-identity features, including a label-free house-number encoding that transfers to
  unseen French words.
- **Cross-encoders:** MiniLM-L12 ×2 (Apache-2.0), multilingual-e5-small ×2 and e5-base (MIT), each
  ≤ 278 M parameters, fine-tuned out of fold on the hard pairs and blended with the matcher.
- **Decision:** every record goes to its best entity; every entity keeps the subset that maximises
  its expected F0.5. Generator-aware rules handle the decoys (house numbers shifted by +K,
  business-type word swaps).
- **France (no labels):** each French rule is validated on a labelled US / India analogue, and on
  label-free test checks (match-count budget, house-number mirror, word-frequency test, level
  matching against training truth).

Full write-up: [METHODOLOGY.md](METHODOLOGY.md).

## Results

| Version | Change | Leaderboard |
|---|---|---|
| v3 | word-identity features, decoy weighting | 0.982 |
| v11 | cross-encoder blend, +K and French word rules | 0.987856 |
| v12e | second-pass search, filter before the matcher, team stack | 0.990159 |
| **v12i** | seed bag, 5th cross-encoder, French level matching | **0.990336** |

Out of fold on train (US / India): 0.99073 / 0.99130.

## Repository layout

```
├── METHODOLOGY.md                    methodology write-up (submitted documentation)
└── code/business_entity_resolution/
    ├── README.md                     step-by-step reproduction, timings, source-file roles
    ├── requirements.txt              pinned dependencies (Python 3.11)
    ├── src/                          the pipeline (entry point: run_all.py)
    └── team_inputs/README.md         the team-mate submission files used by the final stage
```

## Reproducing

1. Put the challenge data in `dataset/` at the repository root (`dataset/train/train_source{1,2,3}.tsv`,
   `train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv`). The data is the organisers' and
   is not included here.
2. Install the requirements: `pip install -r code/business_entity_resolution/requirements.txt`
   (the CUDA build of torch is noted in the file).
3. Run `python code/business_entity_resolution/src/run_all.py`. Outputs go to `output/`, and
   intermediate files to `work/` (≈75 GB at peak, ≈20 h on 14 cores and a 4 GB GPU).

For the leaderboard file, also set `ER_TEAM_FILES` to the three team-mate files. Those files are in
the submission package, not in this repository; see
[code/business_entity_resolution/README.md](code/business_entity_resolution/README.md).

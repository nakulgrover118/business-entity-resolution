# Business Entity Resolution — reproducible pipeline

Two-pass blocking (TF-IDF search, then a wide search for the records the first pass leaves unmatched)
+ learned candidate filter + two-stage LightGBM matcher + multilingual cross-encoder re-scoring of the
hard pairs + LightGBM blend + expected-F0.5 decision rule with generator-aware decoy rules.
Only the provided training/test files are used (no external data or APIs). Models:
LightGBM (MIT); cross-encoders fine-tuned from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(Apache-2.0, 118 M parameters), `intfloat/multilingual-e5-small` (MIT, 118 M parameters) and
`intfloat/multilingual-e5-base` (MIT, 278 M parameters; e5-small is fine-tuned twice, CE3 and CE5), all far below the 8 B limit. Their pre-trained weights are downloaded from the Hugging Face hub
on first use (then cached).

## Environment

* Python 3.11 (tested on Windows 11, 16 GB RAM, 14 cores, one 4 GB NVIDIA GPU for the cross-encoders; ≈120 GB free disk)
* `pip install -r requirements.txt`

## Data layout

The pipeline expects the challenge folder layout:

```
student_resource/
├── dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
├── dataset/test/test_source{1,2,3}.tsv
└── code/business_entity_resolution/src/   (this code)
```

`ER_DATA_ROOT` can point to a different folder containing `dataset/`.
Intermediate files go to `<root>/work/cache` (≈25 GB on disk at peak),
outputs to `<root>/output`. `ER_N_JOBS` sets the number of threads.

## Run end to end

```bash
cd code/business_entity_resolution/src
python run_all.py
```

The pipeline runs in **two passes**. Pass 2 re-runs the pipeline with a much wider search for the query records
that pass 1 could not match. 99% of the true pairs missed by the first blocking belong to such records.

**Pass 1** (cache `work/cache`):

| step | command | what it does | time* |
|---|---|---|---|
| 1 | `python translit.py` | learns the Indic-script → Latin word dictionary from train pairs | 1 min |
| 2 | `python prep.py train test` | normalises names / addresses of every record (parallel) | 6 min |
| 3 | `python records.py train test` | per-country arrays: state removal, city, house number, token ids, IDF | 10 min |
| 4 | `python pairs.py train` / `python pairs.py test` | TF-IDF blocking (passes 1+2) + pair features | 40 min |
| 5 | `python wordte.py` | word-identity features (extra / dropped name words) | 5 min |
| 6 | `python train.py 2200000 2.0` | two-stage LightGBM "v3" (decoy weight 2), out-of-fold `pB_v3.npy` | 45 min |
| 7 | `python predict.py` | test probabilities `pB_v3_test.npy` | 20 min |
| 8 | `python crossenc.py select` / `train 0 3000000` / `train 1 3000000` / `score`, once per round with the environment of `run_all.CE_ROUNDS` (ce1, ce2, ce3, ce4; ce4 is trained but not scored in pass 1) | multilingual cross-encoders fine-tuned on the hard pairs (2 folds by S1, out-of-fold scores) | ≈1–4 h per round (GPU) |
| 9 | `python expand.py <work>/pass2/cache` | CE1–CE3 blend + decision rules; writes masks of the query records left unmatched | 15 min |

**Pass 2** (`ER_WORK_DIR=<work>/pass2`, run by `run_all.py --pass2`; the pass-1 upstream files are hard-linked):

| step | command | what it does | time* |
|---|---|---|---|
| 10 | `python pairs.py train` / `python pairs.py test` | Blocking plus the pass-3 wide search for the masked records: combined-score top-20 with index features up to an S1 document frequency of 20,000 (instead of 4,000), address top-5 and name×city/house-number top-3. Train blocking pair recall: US 98.42% → **99.07%**, India 97.70% → **98.65%**. | 2.5 h |
| 11 | `python prune.py train` then `python prune.py score` | Learned candidate filter: a LightGBM on 45 cheap blocking/lookup features, out of fold on train. | 25 min |
| 12 | `python filter_pairs.py` | Keeps only the pairs above the per-country threshold, **before any matcher runs**. Test ≈62 → **4.4 candidates per S1** (US 4.19, India 4.50, France 4.71); train 4.1–4.4. | 5 min |
| 13 | `python wordte.py` | Word features on the survivors. Label-free rates come from first-pass pairs only, so they do not shift between train and test. | 5 min |
| 14 | `python train.py 2200000 2.0` | v3 trained on the filtered candidates. OOF stage B: US 0.9902, India 0.9905, slightly above training on the unfiltered set. | 12 min |
| 15 | `python predict.py` | Test probabilities on the survivors. | 6 min |
| 15b | `ER_SEED=7 ER_MODEL_TAG=s7 python train.py 2200000 2.0` + `predict.py`, the same for seed 13, then `python bag.py` | **Seed bag:** two extra v3 seeds, logit-averaged with the original (seed 42) into `pB_bag*.npy`. Used by the US/India blend; France keeps the original v3. | 40 min |
| 16 | `python crossenc.py select` / `score` per round, with `CE_REUSE` pointing at the pass-1 round | The pass-1 cross-encoders CE1–CE4 score the pass-2 hard pairs; identical pairs are reused. | ≈1 h (GPU) |
| 16b | `python crossenc.py select` / `train 0 3000000` / `train 1 3000000` / `score` with the `ce5` environment of `run_all.CE_ROUNDS` | **CE5:** multilingual-e5-small fine-tuned on the pass-2 hard pairs (2 epochs, 2 folds). These include wide-search pairs that CE1–CE4 never saw in training. | ≈2 h (GPU) |
| 17 | `python final.py` | LightGBM blend of the bagged v3 + CE1–CE5 + pair features + the signed house-number offset (US/India), with 3 blender seeds averaged; +K penalty (India); France: original v3 + word rules; expected-F0.5 decision → `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The candidate file is exactly the set the matcher scored. | 25 min |
| 18 | `python team_stack.py ...` (or `ER_TEAM_FILES="..;..;.." python run_all.py`) | **Team ensemble and French rules (leaderboard file).** See below. The candidate file gets the pairs added by this stage appended, so every match is a candidate. Regenerates the leaderboard file byte-for-byte (**v12i, 0.990336**). | 15 min |

\*on the machine above (14 cores, 16 GB RAM, 4 GB GPU). The total is ≈19–21 h, and pass 2 needs ≈75 GB of disk.

**Step 18 in detail** (`team_stack.py`, inputs in `team_inputs/`):

1. **US / India stack** with team files 1 and 2. Disputed pairs are judged by our blend probability (our candidates) or the calibrated CE3 score (other pairs). Additions need p ≥ 0.5 (not +K) or CE3 ≥ 0.7; removals are only for +K decoys.
2. **French initials refinement** with team file 2.
3. **Team file 3's French additions**, kept only in two label-supported classes. Each class is validated on a labelled US/India analogue:
   - a filler word added or swapped in at the same house number (98.5–100% true);
   - identical name tokens, no address, and the S1 name unique in France (97–98% true).
4. **France level matching.** France has no training labels, so each French edit class is compared with the share that is true in training:
   - team additions beyond the train-truth share of their class are dropped (lowest v3 probability first), as are filler swaps that drop the legal form;
   - "S1 name + *développement* / *groupe* / *federation*" at the same house number with the same legal form is added. It is 99–100% true in training, but our French matcher assigned only 11% / 81% / 0% of it.

Every step keeps each record matched to at most one S1.

Out-of-fold macro F0.5 (train, US / India):

| model | US / India |
|---|---|
| v3, pass 1 | 0.9865 / 0.9861 |
| + CE1–CE3 blend, pass 1 (v11) | 0.98911 / 0.98885 |
| v3, pass 2 | 0.99007 / 0.99016 |
| + CE1–CE4 blend, pass 2 (unfiltered) | 0.99049 / 0.99127 |
| + signed house-number offset and pair features in the blend (`base2`) | 0.99079 / 0.99132 |
| filter before the matcher, v3 retrained on the survivors | 0.99073 / 0.99127 |
| **final: + seed-bagged v3, CE5, 3 blender seeds** | **0.99073 / 0.99130** |

Leaderboard: v12e (stack, without the France step 18.3–18.4 and without the bag / CE5) 0.990159 → **v12i 0.990336**.

`python blend2.py eval ce1,ce2,ce3,ce4,base,base2` prints the out-of-fold score of the blend (`ER_V3_TAG=bag` for the bagged v3).

## Outputs

* `output/matching_results.tsv` — one row per test Source 1 entity, matched S2/S3 ids.
* `output/candidate_pairs.tsv` — the exact candidate set scored by the model: the learned-filter survivors
  of our two-pass blocking (the set that v3, the cross-encoders and the decision run inference over), plus the
  10,953 pairs (0.14 %) added by the team stage (`team_stack.py`: pairs proposed by the team-mate submissions
  and judged by our calibrated CE3 or the French rules), so every match is a candidate. 4.42 candidates per S1.

Validate with `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` from `student_resource/`.

## Source files

| file | role |
|---|---|
| `config.py` | paths, thread count, seed |
| `io_utils.py` | TSV loading (cached as Parquet), ground truth |
| `translit.py` | learned Indic-script → Latin token / state-name dictionary |
| `normalize.py` | name & address normalisation (aliases, websites, legal forms, abbreviations, ordinals, leet typos) |
| `prep.py` | parallel normalisation of all records |
| `records.py` | data-driven state vocabulary + canonical states, city / house number, token CSR + IDF, word segmentation of concatenated names |
| `blocking.py` | sparse TF-IDF candidate generation (name, bigrams, address, combos) + address-only pass + the pass-2 wide search for unmatched records |
| `features.py` | pair features (numba token alignment, rapidfuzz similarities, legal forms, context) |
| `pairs.py` | blocking + features → disk (memory-mapped) |
| `wordte.py` | word-identity features: supervised out-of-fold and label-free (house-number agreement) encodings of added / dropped name words |
| `model.py` | LightGBM stages, stage-B competition features |
| `train.py` | 2-fold training, OOF evaluation, model saving |
| `decision.py` | one-to-many constraint + expected-F0.5 optimal selection |
| `evaluate.py` | macro F0.5 exactly as defined by the challenge |
| `predict.py` | test inference (v3), candidate file |
| `crossenc.py` | multilingual cross-encoder fine-tuning / scoring on the hard pairs |
| `blend2.py` | LightGBM blend of v3 + cross-encoder scores (+ margins and a few pair features), out-of-fold evaluation |
| `prune.py` | learned blocking filter (candidate pruning) and its per-country thresholds |
| `filter_pairs.py` | applies the filter to the candidate files before the matcher (the candidate set fed to the model) |
| `bag.py` | logit-average of the v3 seeds (seed bag) |
| `team_stack.py` | team ensemble: stack with the team-mate submissions (`team_inputs/`) + French refinement, team additions and level matching → leaderboard file |
| `final.py` | final decision: CE1–CE5 blend on the bagged v3 (3 blender seeds), logit shifts, +K house-number decoy rule, France word rules, candidate filter, output files |
| `expand.py` | masks of the query records left unmatched by pass 1 (input of the pass-2 wide search) |
| `run_all.py` | runs both passes in order (`--pass2` runs pass 2 only) |

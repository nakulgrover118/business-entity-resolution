# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Gym Bros
**Team Members:** Nakul Grover, Manjot Singh, Arshpreet Gill
**Submission Date:** 2026-09-28

---

## 1. Executive Summary

We resolve Source 2/3 records against the deduplicated Source 1 reference with a classic
**blocking → pairwise classifier → global decision** pipeline, engineered around what the data
generator actually does. Blocking is a sparse IDF-weighted similarity search (names, name bigrams,
addresses, house-number/street and name×city combos) with an address-only pass. It runs in **two
passes**: the records the first pipeline leaves unmatched get a much wider second search. A learned
cheap-feature filter then cuts the candidate set to **≈4.4 per Source-1 entity**. A
**two-stage LightGBM** model scores candidate pairs; stage B adds "competition" features that
encode the fact that every S2/S3 record belongs to at most one S1 entity. The final match list of
each entity is the subset that **maximises the expected F0.5** under the predicted probabilities
(which also decides when to predict "no match"). The hard pairs are then re-scored by fine-tuned
**multilingual cross-encoders** (2× MiniLM-L12, Apache-2.0; e5-small and e5-base, MIT; 118–278 M parameters) and a
LightGBM blender; generator-aware rules handle the decoys that dominate the test set (house numbers
shifted by +K, one-word "business type" swaps in France). Out-of-fold macro F0.5 on the training
data: **0.9862** (v3 LightGBM) → 0.9890 (blend) → **0.9909** (second-pass search + four cross-encoders).
Blocking pair recall is 99.07 % (US) / 98.65 % (India). Leaderboard: 0.967 (v1) → 0.973 → 0.9828 → 0.9851
(cross-encoder blend) → 0.987856 (decoy rules) → 0.989829 (second-pass search, stacked with a team-mate's
submission) → 0.990159 (filter before the matcher, signed house-number offset in the blend) → **0.990336**
(seed-bagged matcher, a fifth cross-encoder, and French edit classes level-matched against training
truth). All features are country-agnostic, so France (unseen in training) uses the same model plus
label-free rules, each validated on a labelled US/India analogue.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA)

* **Scale:** train 2.2 M S1 / 5.0 M S2 / 5.3 M S3; test 1.73 M / 4.89 M / 5.08 M
  (US 38 %, India 47 %, France 15 % of test S1).
* **Cardinality:** every S2/S3 record matches **at most one** S1 entity (7.64 M matched ids, all
  unique). ~27 % of S2/S3 records match nothing (distractors). 5.6 % of S1 entities are singletons;
  others have 1–11 matches (mean ≈ 3.5).
* **Name noise:** legal-suffix drops/abbreviations/reordering (Pvt ↔ Private, Inc ↔ Incorporated,
  "LLC Dalton…"), filler words (Center, Services, Group…), word duplication, typos incl. `�`
  replacement characters, accents and leet digits (`S0lutions`, `8lack`), brackets and decorations
  (`>>`, `***`, `(ID: 8283)`, `| www.x.com`), website/handle names (`modernconstructionenterprises.com`,
  `#alhigh`, dot-less `GSVISIONCOM`), alias names `<FakeBrand> dba|f/k/a|née|trading as… <Original>`
  (original always after the phrase), fully replaced fake brands (`Beloumbra`), and for India names
  transliterated into 9 Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali,
  Malayalam, Odia, Gurmukhi), sometimes partially.
* **Address noise:** upper-casing and abbreviations (ST/RD/AVE, R./BD/AV/IMP for France), component
  reordering (state first/middle), dropped components (truncated to "no, city"), empty addresses,
  PO Box / PMB / unit insertions, house-number perturbations (`02243`, `3977-3979`, `6-4 5 720`),
  varying prefixes (Door No / H.No / Flat No), city variants (`CHICAGO CITY`, `LAKE BLUFF CDP`,
  `Bengaluru`/`Bangalore`), and state variants: US codes vs full names, India full names vs codes
  vs native script, **France regions (S1) vs departments (S2/S3)**.
* **Hard negatives:** many distractors are *siblings* of S1 entities — same address with one name
  word changed (`RS Merchants Center` → `RS Creations Center`), a changed legal form
  (`Pvt Ltd` → `Public Limited`/`LLP`), or a shifted house number. True matches can also have
  shifted numbers, so these are partly irreducible.
* **Legal-form transitions are structured:** for true matches a suffix is kept, dropped, abbreviated
  or (when S1 has none) a generic one is added (Inc/LLC/Co/Ltd), PC/LP may become Inc/LLC; conflicts
  such as `LLC→Inc` or adding `Pvt Ltd` essentially never occur for true matches but are common
  among distractors (legal conflict: 0.03–0.5 % of positives vs 15–19 % of negatives).

### 2.2 Solution Strategy

**Approach type:** Blocking + two-stage gradient-boosted classifier + expected-F0.5 decision (hybrid).

**Core innovations**
1. Learned, data-driven normalisation: Indic → Latin word dictionary and native-script state names
   learned from training pairs; state vocabulary and state canonicalisation (e.g. `gironde →
   nouvelle aquitaine`, `texas → tx`, `mh → maharashtra`) learned *unsupervised* from city
   co-occurrence, so it works for France without labels.
2. Blocking with name/address/combination features and an address-only rescue pass.
3. Stage-B competition features exploiting the one-S1-per-record constraint.
4. Exact expected-F0.5 subset selection per entity (Poisson-binomial DP), which also decides
   singletons.
5. Word-identity features with a **label-free** house-number-agreement encoding that transfers to
   French words never seen in training.
6. Cross-encoder re-scoring of only the ~5 % "hard" pairs, blended with the tree model out of fold.
7. Decoy rules reverse-engineered from the generator and verified on labels: +K house-number
   shifts (US/India) and business-type word swaps vs filler-word swaps (France).
8. Label-free validation for the unlabelled country (France). Three tools, each calibrated on
   labelled US/India data:
   - a match-count budget: the true matches-per-S1 distribution is identical across countries;
   - a word test: how often a word appears in S1 names vs S2/S3 names, which separates noise words
     from ordinary name words;
   - level matching: a French edit class is added or trimmed until its assigned share equals the
     share that is true for the same class in training.

---

## 3. Candidate Generation (Blocking)

Per country (country is only used as a partition key, never as a closed set), every record is
turned into a sparse vector with four L2-normalised groups mixed with weights
0.40 / 0.35 / 0.25:

* **Name group:** core name tokens (legal forms/honorifics removed, transliterated, concatenated
  web names segmented into dictionary words with a unigram DP), adjacent-token bigrams, alias/website
  tokens (half weight). Weights = IDF.
* **Address group:** address tokens with the state removed, plus *house number + first
  non-common word* keys (skips `rue`/`de`/`st` so it works for French "8 Rue de la Paix").
* **Combination group:** name token × city and name token × house number, which separates generic
  names (`Green Consulting`, `Galaxy Services`) that exist dozens of times in a country.

Features with Source-1 document frequency > 4 000 are removed from the index only (norms keep
them) to bound the cost of the multithreaded sparse top-n product (`sparse_dot_topn`).

* **Pass 1:** for every S2/S3 record the top-10 S1 records with score ≥ 0.6 × its best score.
* **Pass 2 (address-only):** the top-1 S1 record by address cosine (≥ 0.3), which recovers records
  whose name was replaced by a filler/fake brand but whose address is intact.

| | India train | US train | test (all) |
|---|---|---|---|
| candidate pairs | 11.9 M (13.5 / S1) | 14.2 M (10.7 / S1) | 28.5 M (16.5 / S1) |
| pair recall | 97.70 % | 98.42 % | – |
| macro F0.5 ceiling | 0.9927 | 0.9950 | – |

**Keeping true matches:** 96.5 % of true matches are the #1 candidate of their record; the
top-10/relative cut-off keeps 99.7 % of what an unrestricted top-10 finds while using 4× fewer
pairs; the address pass adds +0.2 % recall.

**Second pass for unmatched records.**
- **Target:** 99 % of the true pairs the first blocking misses belong to query records that the
  complete first pipeline (matcher + cross-encoders + decision) leaves unmatched: 28 % of train and
  41–43 % of test records.
- **The wide search** for those records only:
  - top-20 on the combined score **without** the relative cut, with index features kept up to an S1
    document frequency of **20 000** instead of 4 000;
  - address-only top-5;
  - name×city / name×house-number top-3.
- **Why keeping common words matters:** at the old limit the same top-20 recovers only 23 % of the
  misses; at 20 000 it recovers 36 %. Common words and city names carry the signal for records with
  replaced names or truncated addresses.
- **Result:** 41 % of the missed true pairs are recovered in both countries. The pipeline is then
  re-run on the enlarged candidate set, with a `blk_expanded` flag feature and word statistics taken
  from first-pass pairs only, so they do not shift between train and test. The learned filter below
  runs before the matcher, so the matcher trains on all surviving pairs.

| | India train | US train | test |
|---|---|---|---|
| candidates after pass 2 (before filtering) | 34.2 M (38.7 / S1) | 47.9 M (36.2 / S1) | 108 M (≈62 / S1) |
| pair recall | 97.70 % → **98.65 %** | 98.42 % → **99.07 %** | – |

**Learned candidate filter (`prune.py`).**
- **Model:** a LightGBM (300 trees, 2 folds by S1) on 45 cheap features only (blocking scores and
  ranks, IDF cosines, token alignment, house-number / city / state / legal-form lookups, empty-address
  and source flags). It scores every candidate pair; pairs below a per-country threshold are dropped
  before the expensive matcher.
- **Test:** ≈62 → **4.4 candidates per S1** (US 4.19, India 4.50, France 4.71).
- **Out-of-fold F0.5 change:** −0.00003 (US) and −0.00008 (India). About 1 000 of 5.8 M test matches are lost.
- **Remaining misses** are mostly records with an empty address *and* a generic name shared by several
  S1 entities. They cannot be resolved.

**Submitted `candidate_pairs.tsv` (4.42 candidates per S1):**
- the filter survivors above, which is exactly the set that the v3 matcher, the cross-encoders and the
  decision run inference over;
- plus 10,953 pairs (0.14 %) added by the team stage (section 4.3): pairs proposed by the team-mate
  submissions and judged by our calibrated CE3 or the French rules. They are included so that every
  match is a candidate.

---

## 4. Matching Model

**Features (76 pair features + 12 word-identity features in both stages + 12 stage-B features, all country-agnostic):**

* *Name:* IDF-weighted token alignment in both directions with typo-tolerant token matching
  (Jaro-Winkler ≥ 0.86 or prefix/abbreviation): hard and soft coverage, max IDF and count of
  unmatched tokens, exact-token share, first-token similarity, token counts, IDF sums, IDF cosine;
  rapidfuzz ratio, token-set, token-sort, partial ratio, Jaro-Winkler, concatenated-name ratio and
  partial ratio, full-name (with legal form) ratio; alias/website coverage of the S1 name.
* *Address:* alpha-token alignment (same statistics as names), IDF cosine, numeric-token coverage
  both ways, house-number equality, relative difference, digit-substring relation and digit
  edit distance, city equality and Jaro-Winkler, canonical-state equality, rapidfuzz token-set /
  ratio / partial ratio of address strings, empty-address flag.
* *Legal form:* same / Jaccard / conflict / public-mismatch / generic relation code
  (equal, dropped, added, subset, superset, disjoint) and categorical legal-family ids of both
  sides (ids unseen in training — e.g. French SARL/SAS — are mapped to missing; 15 % random masking
  during training teaches the model to fall back on the generic features).
* *Word identity (v3):* for every name word of the S2/S3 record without a counterpart in the S1 name
  ("extra" word) and every S1 word missing from the record ("dropped" word): (a) its out-of-fold match
  rate among near-identical training pairs (sibling decoys add branch-like words such as
  *group, holdings, industries, north, westgate, midtown* — match rate 0.0 — whereas true-match noise
  adds fillers such as *center, services* — ~0.2); (b) a **label-free** score computed on each split's
  own pairs: how often the house number still agrees when this word is added/dropped (siblings move
  the house number, noise fillers usually keep it; correlation with the supervised match rate 0.86).
  (b) transfers to unseen languages: French sibling markers *groupe, holding, développement,
  participations, distribution, international* get 6–8 % agreement vs 41 % for *services*. The
  supervised encodings are hidden on 30 % of training rows so the model also learns to rely on (b).
* *Record flags:* native script, pipe/website, alias phrase, web name, unknown native tokens, source.
* *Blocking context:* score, rank / count / gap to best / second best within the record's list and
  within the entity's list.
* *Stage B (from out-of-fold stage-A probabilities):* the pair probability, best competing S1
  probability for the same record and the margin to it, rank within record and entity, number and
  sum of other confident candidates of the entity (overall and from the same source).

**Model type:** LightGBM binary (255 leaves, lr 0.06, feature/bagging fraction 0.7/0.8, early
stopping) — MIT licence, a few MB, far below the 8 B-parameter limit. Two folds split by S1 entity
give out-of-fold probabilities for every training pair; stage-B training uses those; test
predictions average both fold models.

**Decision / threshold selection:** each S2/S3 record is only offered to its highest-probability
S1 entity; for every S1 entity the top-k candidates maximising the expected F0.5 (exact
Poisson-binomial computation, k = 0 allowed) are output. This beat every global threshold on the
out-of-fold data.

### 4.1 Cross-encoder re-scoring and blend (US / India)

* **Hard pairs:** for every S2/S3 record its top-2/top-3 candidates whose v3 probability lies in
  [0.002, 0.998] (≈5 % of candidate pairs; 0.5–0.9 M per split and country).
* **Cross-encoders:** the raw "name | address" texts of both records are read jointly by a small
  multilingual transformer fine-tuned as a binary classifier (fp16, frozen embeddings, max length 128,
  2 folds by S1 entity so every training pair gets an out-of-fold score; test = mean of both folds).
  Rounds: CE1 = MiniLM-L12 (1 epoch, top-2), CE2 = MiniLM-L12 (2 epochs, top-3, other seed),
  CE3 = multilingual-e5-small (2 epochs, top-3), CE4 = multilingual-e5-base (1 epoch, top-3, batch 24
  with fp16 frozen embeddings on a 4 GB GPU). CE1–CE4 are trained on the first-pass hard
  pairs and score the second-pass hard pairs (identical pairs reuse their scores). CE5 =
  multilingual-e5-small (2 epochs) is trained on the second-pass hard pairs themselves, which include
  wide-search pairs that CE1–CE4 never saw.
* **Seed bag:** the v3 matcher is trained with three seeds (42, 7, 13); their logits are averaged.
  The bag feeds the US/India blend.
* **Blend:** a LightGBM (500 rounds, 63 leaves) on the hard pairs with the v3 logit and its margin to
  the record's best competitor, each cross-encoder score and its margin, and 8 pair features
  (label-free word/house-number encodings, legal relation, name/address cosine, source, empty
  address). Out of fold: US 0.9865 → 0.9891, India 0.9861 → 0.9889 (CE1+CE2+CE3).
  Final blend: the bagged v3, CE1–CE5, 23 pair features and the signed house-number offset (with +K / −K
  flags). It is trained with 3 LightGBM seeds (2 folds × 3 models, predictions averaged).
* **France keeps the v3 probabilities:** the cross-encoders learned from US/India that a swapped
  business-type word signals a decoy, but they cannot tell French fillers from French type words
  and re-admitted French decoys.

### 4.2 Final decision rules

* **Strictness:** logit − 0.25 (US/India) and − 1.0 (France), tuned out of fold with decoy false
  positives counted twice (the test set has ~2× more sibling decoys).
* **+K house-number decoys (India −1 on the logit):** the generator makes decoys by
  shifting an S1 house number *upwards* by K ∈ {3, 4, 5, 7, 9, 11, 13, 21}; true copies change
  digits symmetrically. Accepted +K pairs were only 69–78 % true on train.
  - Evidence: a mirror test (+K vs −K counts on test) confirms the excess, and shows none for France.
  - US: since the final blend sees the signed house-number offset itself, it needs no extra penalty
    (0, tuned out of fold; the earlier CE1–CE3 blend used −2).
* **France word rules (one-word name swaps):** a real French word (≥ 4 letters, document
  frequency ≥ 100) swapped in that is not a generator filler (école, club, amicale, comité, sportive,
  amis, primaire…) means a different business → logit − 5 (−10.7 K matches). On train the same
  pattern is 0–5 % true (India *solutions, industries, foundation*…), fillers 95–99 %. Filler swaps
  (groupe, services, développement, france, compagnie/cie, fils, associés, st) with the same house
  number and address → logit + 2 (+2.2 K matches).

### 4.3 Team ensemble and French level matching (final step, `team_stack.py`)

* **US / India:** decision-level stack with two team-mate submissions.
  - Disputed pairs are judged by our blend probability, or by the isotonic-calibrated CE3 score for
    pairs outside our candidates.
  - Additions need p ≥ 0.5 and no +K shift, or CE3 ≥ 0.7. Removals are only for +K decoys.
* **France, with no labels, gets three label-validated edits.** In each, a record is matched to at
  most one S1.
  1. **Initials refinement:** records whose name is the initials of the S1 name, at the same house
     number (99 % true on train).
  2. **Team-mate audit:** a third team file from a label-free France audit. Only its additions in two
     classes are kept:
     - a filler word added or swapped in at the same house number (train analogue 98.5–100 % true);
     - identical name tokens, record without address, S1 name unique in France (train analogue 97–98 %).
  3. **Level matching.**
     - *Why:* France predicted 3.31 matches per S1 against a true 3.46. The true distribution is
       identical in US and India train, and US/India test predicts 3.39.
     - *Word test:* a noise word the generator adds to copies is rare in S1 names relative to S2/S3.
       *développement* 0.09 and *groupe* 0.21, while ordinary French name words are about 1.1
       (*service* 1.13).
     - *House-number fingerprint:* same house vs +K shift, per added word.
     - *Added:* "S1 name + développement / groupe / federation" at the same house number with the
       same legal form. This is 99–100 % true in train (US fillers, including the mixed-use
       *partners*, and org words such as *federation*), but France assigned only 11 % / 81 % / 0 %:
       +1,410 pairs.
     - *Trimmed:* where the team additions pushed France's assigned share above the train truth
       share, the lowest-probability additions were removed. The cells are "record without address,
       identical name": legal form dropped 97.4 % vs 93.4 % true; same legal form 99.6 % vs 99.0 %.
       Filler swaps that drop the legal form (≈2 % true on train) were also removed.

---

## 5. Results & Error Analysis

| | India | US | overall |
|---|---|---|---|
| Stage A OOF macro F0.5 | 0.9841 | 0.9846 | 0.9844 |
| Stage B + expected-F0.5 | 0.9858 | 0.9864 | **0.9862** |

(First version without the v2 improvements: 0.9811 / 0.9841 / 0.9829.)


**Distribution shift and v3.** The test set contains ~2× more sibling decoys per entity than the
training set (S2+S3 records per S1: 5.8 vs 4.7) and 2.5–4× more ambiguous pairs; this explained the
gap between out-of-fold (0.986) and leaderboard (v1 0.967, v2 0.973). v3 adds the word-identity
features above and up-weights decoy negatives ×2 during training to match the test prior; on the test
set it reduces the number of ambiguous (0.1–0.9) best pairs per entity by 42 % (US), 57 % (India) and
26 % (France), with out-of-fold macro F0.5 0.9863 (0.9858 with decoy false positives counted twice).

**Leaderboard history (public test):**

| version | change | score |
|---|---|---|
| v1 | first pipeline | 0.967 |
| v2 | normalisation, address pass, legal-form ids | 0.973 |
| v3 | word-identity features, decoy weight 2 | 0.982 (France −1.0: 0.9828) |
| v6 | + cross-encoder CE1 blend (US/India) | 0.9851 |
| v11 | CE1+CE2+CE3 blend, +K rule, France word rules | 0.987856 |
| v11 + team-mate stack | decision-level stack with a team-mate's submission | 0.989288 |
| v12b + team-mate stack | second-pass wide search, retrained matcher, CE1–CE4 blend | 0.989829 |
| v12c | + second team file stack, French initials refinement | 0.989994 |
| v12e | learned filter **before** the matcher (v3 retrained on ≈4.4 candidates / S1), blend with the signed house-number offset and 23 pair features | 0.990159 |
| v12e, France from the CE blend (A/B) | cross-encoder blend also for France | 0.989973 (France worse) |
| **v12i (final)** | seed-bagged v3, CE5, 3 blender seeds (US/India); team-mate France audit + level matching and filler additions (France) | **0.990336** |

**Out-of-fold macro F0.5 (train, US / India):**

| model | US | India |
|---|---|---|
| v3 (pass 1) | 0.9865 | 0.9861 |
| + CE1–CE3 blend (v11) | 0.98911 | 0.98885 |
| v3 (pass 2) | 0.99007 | 0.99016 |
| + CE1–CE4 blend (v12b) | 0.99049 | 0.99127 |
| + candidate filter, v3 retrained on the survivors, signed offset in the blend (v12e) | 0.99073 | 0.99127 |
| + seed-bagged v3, CE5, 3 blender seeds (final) | 0.99073 | 0.99130 |

* **Common false positives:** generator-made siblings — identical name with a house number shifted
  by a few units, same name and address with a changed legal form, filler-word substitutions
  (`X Services` vs `X Finance` at the same address), and records with an empty address whose
  generic name exists for several S1 entities.
* **Common false negatives:** true matches with heavily corrupted names *and* perturbed numbers,
  empty-address records with generic names, fully replaced fake-brand names with truncated
  addresses, and true matches whose house number was shifted like a sibling's.
* **Structural floor:** records without an address are 97.7 % true copies in train, at the same
  per-S1 rate in test.
  - Most of the remaining recall loss in every country is namesakes: records whose name is shared by
    several S1s, or matches no S1 exactly.
  - For these, the true S1 is often not even among the candidates (train out-of-fold: 63–78 %), and
    the name alone cannot separate them.
  - Test behaves exactly like train out of fold here, so this is a floor rather than a calibration
    problem.
* **France:** no labels.
  - The match-count budget showed France ≈0.08 matches per S1 short of US/India. The level-matching
    step (4.3) recovers the part that label-validated evidence supports.
  - Correct handling of French abbreviations, departments vs regions, website and alias names, and
    rejection of sibling records were checked by hand.
  - A leaderboard A/B showed that the cross-encoder blend hurts France (−0.0012 on France), so France
    keeps the v3 matcher + rules.

---

## 6. Conclusion

Careful normalisation learned from the data (transliteration, state canonicalisation), a
recall-oriented but compact blocking, pairwise features that mirror the generator's noise, and a
decision rule that directly optimises the per-entity F0.5 give a macro F0.5 of about 0.9862 out of
fold. A second, much wider search for the records the first pipeline leaves unmatched, plus cross-encoder
re-scoring of the hard pairs, raise it to ≈0.9910, with ≈4.4 candidates per S1 after the learned filter.
The final leaderboard score is **0.990336**.

The biggest lessons:
- exploit the one-to-one-per-record structure (stage B);
- treat singletons through expected F0.5 rather than thresholds;
- keep every feature country-agnostic, so an unseen country works out of the box;
- for the unseen country, compare each decision class with its labelled analogue before trusting
  the model's confidence.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its README. Entry point `src/run_all.py` runs two passes.

- **Pass 1:** `translit.py` → `prep.py` → `records.py` → `pairs.py` → `wordte.py` → `train.py` →
  `predict.py` → `crossenc.py` (4 rounds) → `expand.py` (masks of the unmatched records).
- **Pass 2:** `pairs.py` (with the wide search) → `prune.py` → `filter_pairs.py` → `wordte.py` →
  `train.py` / `predict.py` (seeds 42, 7, 13) → `bag.py` → `crossenc.py` (CE1–CE4 select/score, CE5
  select/train/score) → `final.py`.

Pass 2 writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The team ensemble
(`team_stack.py`, three team-mate files in `code/business_entity_resolution/team_inputs/`) then stacks
the US/India decisions and applies the French edits (4.3). It regenerates the leaderboard file (0.990336)
byte-for-byte. The pairs it adds are appended to the candidate file, so every match is a candidate.

**Validation discipline.**
- **Choices:** every modelling choice is made on out-of-fold scores (folds by S1 entity) with a fixed adoption bar (≥ +0.0001 weighted, no loss in either country or metric, consistent across folds). The public leaderboard is used only to confirm.
- **Rejected ideas:** the rejected experiments are logged: blender variants, isotonic calibration, and the cross-encoder blend for France (a leaderboard A/B: −0.0012 on France).
- **Below the bar individually, adopted combined:** v3 seed bagging (+0.000016) and CE5 (+0.000005) each fell below the bar. Combined with 3 blender seeds they give +0.00002, consistent in both folds, and went into the final file.
- **France rules without labels:** each French rule was validated on a labelled US/India analogue and on test-side label-free checks (count budget, house-number mirror, word-frequency test, level matching). Rule candidates that failed were not used. For example, extra identical-name records and filler swaps were already at the US/India level, and *service* is an ordinary French name word.

### B. Additional Results

* Transliteration dictionary: 1 347 name tokens + 16 native-script state names; covers 97.7 % of
  native-script name tokens in train and 96.4 % in test (the uncovered 171 words occur only in
  distractor records).
* Blocking experiments on India train (pairs / recall / ceiling): top-10 unrestricted 41.2 M /
  97.60 % / 0.9924; top-10 with relative cut-off 11.0 M / 97.43 % / 0.9918; + character 4-grams
  12.4 M / 97.49 % (no gain, dropped); + address pass (top-1) 11.9 M / 97.68 % / 0.9927.

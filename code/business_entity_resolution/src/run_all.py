"""End-to-end driver, two passes.

Pass 1 (cache <work>/cache): data -> normalisation -> records -> TF-IDF blocking (passes 1+2) + pair
features -> word features -> LightGBM "v3" -> cross-encoders CE1-CE4 (trained here on the pass-1 hard
pairs) -> CE1-CE3 blend + decision rules -> masks of the query records left unmatched (expand.py).

Pass 2 (cache <work>/pass2/cache, run in a sub-process with ER_WORK_DIR): the same blocking plus a wide
third search for the unmatched records (blocking.generate, expand_q) -> pair features -> learned candidate
filter (prune.py, cheap features, ~4.4 candidates / S1) applied BEFORE the matcher (filter_pairs.py) ->
word features (label-free rates from first-pass pairs only) -> LightGBM v3 on the survivors (+ seeds 7 and
13, logit-averaged by bag.py) -> scoring of the pass-2 hard pairs with the pass-1 cross-encoders CE1-CE4 and
a new CE5 trained on the pass-2 hard pairs -> CE1-CE5 blend on the bagged v3 (+ signed house-number offset
and pair features, 3 blender seeds) + decision rules -> output/matching_results.tsv and
output/candidate_pairs.tsv (exactly the pairs the matcher scored).
With ER_TEAM_FILES="<file 1>;<file 2>;<file 3>" the team ensemble (team_stack.py) then produces the
leaderboard file.

Usage: python run_all.py              (both passes)
       python run_all.py --pass2      (pass 2 only; needs a finished pass 1)
"""
import os
import subprocess
import sys

import config

HERE = os.path.dirname(os.path.abspath(__file__))
PASS2_WORK = os.path.join(config.WORK_DIR, "pass2")

# cross-encoder rounds (see crossenc.py); CE1-CE3 feed the pass-1 decision, CE1-CE4 the final blend
CE_ROUNDS = {
    "ce1": dict(CE_TAG="", CE_TOPK="2", CE_LO="0.003", CE_HI="0.997", CE_EPOCHS="1", CE_SEED="0",
                CE_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "ce2": dict(CE_TAG="r2", CE_TOPK="3", CE_LO="0.002", CE_HI="0.998", CE_EPOCHS="2", CE_SEED="1",
                CE_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "ce3": dict(CE_TAG="r3", CE_TOPK="3", CE_LO="0.002", CE_HI="0.998", CE_EPOCHS="2", CE_SEED="3",
                CE_MODEL="intfloat/multilingual-e5-small"),
    # e5-base on a 4 GB GPU: batch 24, fp16 frozen embeddings, crash-insurance checkpoints
    "ce4": dict(CE_TAG="r4", CE_TOPK="3", CE_LO="0.002", CE_HI="0.998", CE_EPOCHS="1", CE_SEED="4", CE_BS="24",
                CE_LR="3e-5", CE_EMB_HALF="1", CE_PRED_BS="64", CE_CKPT_EVERY="5000",
                PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128", CE_MODEL="intfloat/multilingual-e5-base"),
    # pass 2 only: e5-small fine-tuned on the pass-2 (filtered) hard pairs
    "ce5": dict(CE_TAG="r5", CE_TOPK="3", CE_LO="0.002", CE_HI="0.998", CE_EPOCHS="2", CE_SEED="5", CE_BS="64",
                CE_CKPT_EVERY="5000", CE_MODEL="intfloat/multilingual-e5-small"),
}
BAG_SEEDS = (7, 13)  # extra v3 seeds (the original model uses config.SEED = 42); bag.py averages the three
UPSTREAM = ("translit_dict.json", "legal_vocab.json")  # plus rec_*.pkl, raw_*.parquet, norm_*.parquet


def _crossenc(round_name, args, extra=None):
    env = dict(os.environ, **CE_ROUNDS[round_name], **(extra or {}))
    subprocess.run([sys.executable, os.path.join(HERE, "crossenc.py")] + args, env=env, check=True)


def pass1(max_rows=2_200_000, decoy_w=2.0):
    import expand
    import pairs
    import predict
    import prep
    import records
    import train
    import translit
    import wordte
    translit.build_dictionary()
    prep.main(("train", "test"))
    for split in ("train", "test"):
        records.build_split(split)
    for split in ("train", "test"):
        for country in records.split_countries(split):
            pairs.build(split, country)
    wordte.build_all()
    train.main(max_rows, decoy_w)          # v3 -> pB_v3.npy (out of fold)
    predict.main()                         # pB_v3_test.npy
    for r in ("ce1", "ce2", "ce3", "ce4"):
        for args in (["select"], ["train", "0", "3000000"], ["train", "1", "3000000"]):
            _crossenc(r, args)
        if r != "ce4":
            _crossenc(r, ["score"])        # CE1-CE3 scores feed the pass-1 decision
    os.makedirs(os.path.join(PASS2_WORK, "cache"), exist_ok=True)
    expand.main(os.path.join(PASS2_WORK, "cache"))


def _link_upstream():
    src, dst = config.CACHE_DIR, os.path.join(PASS2_WORK, "cache")
    for f in os.listdir(src):
        if f in UPSTREAM or f.startswith(("rec_", "raw_", "norm_")):
            if not os.path.exists(os.path.join(dst, f)):
                os.link(os.path.join(src, f), os.path.join(dst, f))  # hard link, no extra disk


def pass2(decoy_w=2.0):
    """Runs with ER_WORK_DIR=<work>/pass2 (config.CACHE_DIR is then the pass-2 cache)."""
    import bag
    import filter_pairs
    import final
    import pairs
    import predict
    import prune
    import records
    import train
    import wordte
    for split in ("train", "test"):
        for country in records.split_countries(split):
            pairs.build(split, country)    # uses cache/expand_{split}_{country}.npy
    prune.train()                          # learned candidate filter on cheap features (out of fold on train)
    prune.score("test")
    filter_pairs.main()                    # keep only the survivors: the matcher never sees the rest
    wordte.build_all()
    train.main(2_200_000, decoy_w)
    predict.main()
    for s in BAG_SEEDS:  # extra v3 seeds (sub-processes: the seed is read at import); their outputs go aside
        env = dict(os.environ, ER_SEED=str(s), ER_MODEL_TAG=f"s{s}",
                   ER_OUTPUT_DIR=os.path.join(config.WORK_DIR, f"seed{s}_output"))
        subprocess.run([sys.executable, os.path.join(HERE, "train.py"), "2200000", str(decoy_w)], env=env, check=True)
        subprocess.run([sys.executable, os.path.join(HERE, "predict.py")], env=env, check=True)
    bag.main(("v3",) + tuple(f"s{s}" for s in BAG_SEEDS))
    pass1_cache = os.environ["ER_PASS1_CACHE"]
    for r in ("ce1", "ce2", "ce3", "ce4"):
        d = "crossenc" + (f"_{CE_ROUNDS[r]['CE_TAG']}" if CE_ROUNDS[r]["CE_TAG"] else "")
        for f in (0, 1):  # pass-1 models score the pass-2 hard pairs (reusing identical pairs)
            src, dst = os.path.join(pass1_cache, d, f"model_f{f}"), os.path.join(config.CACHE_DIR, d, f"model_f{f}")
            if not os.path.exists(dst):
                os.makedirs(dst)
                for x in os.listdir(src):
                    os.link(os.path.join(src, x), os.path.join(dst, x))
        _crossenc(r, ["select"])
        _crossenc(r, ["score"], {"CE_REUSE": f"{os.path.join(pass1_cache, d)}|{pass1_cache}"})
    # CE5 is trained here, on the pass-2 hard pairs (same selection rule as CE3 / CE4)
    for args in (["select"], ["train", "0", "3000000"], ["train", "1", "3000000"], ["score"]):
        _crossenc("ce5", args)
    final.main(os.path.join(config.OUTPUT_DIR, "matching_results.tsv"))
    team = os.environ.get("ER_TEAM_FILES", "")
    if team:  # team ensemble (leaderboard file): ER_TEAM_FILES="<team file 1>;<team file 2>;<team file 3>"
        import team_stack
        t1, t2, t3 = team.split(";")
        out = config.OUTPUT_DIR
        team_stack.main(os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv"), t1, t2, t3,
                        os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv"))


def main(max_rows=2_200_000, decoy_w=2.0):
    pass1(max_rows, decoy_w)
    _link_upstream()
    env = dict(os.environ, ER_WORK_DIR=PASS2_WORK, ER_PASS1_CACHE=config.CACHE_DIR)
    subprocess.run([sys.executable, os.path.abspath(__file__), "--pass2"], env=env, check=True)


if __name__ == "__main__":
    if "--pass2" in sys.argv:
        pass2()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 2_200_000,
             float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)

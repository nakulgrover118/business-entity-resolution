"""Paths and global settings for the entity-resolution pipeline."""
import os

# Root of the student_resource folder (contains dataset/). Override with ER_DATA_ROOT.
DATA_ROOT = os.environ.get(
    "ER_DATA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
DATASET_DIR = os.path.join(DATA_ROOT, "dataset")
WORK_DIR = os.environ.get("ER_WORK_DIR", os.path.join(DATA_ROOT, "work"))
CACHE_DIR = os.path.join(WORK_DIR, "cache")
OUTPUT_DIR = os.environ.get("ER_OUTPUT_DIR", os.path.join(DATA_ROOT, "output"))

N_JOBS = int(os.environ.get("ER_N_JOBS", max(1, (os.cpu_count() or 4) - 4)))
SEED = int(os.environ.get("ER_SEED", "42"))  # ER_SEED: extra training seeds for model bagging

SOURCES = ("source1", "source2", "source3")

for _d in (WORK_DIR, CACHE_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)


def tsv_path(split, source):
    return os.path.join(DATASET_DIR, split, f"{split}_{source}.tsv")


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


# Constants
DATA_PATH = "data"
WANDB_PROJECT = "contradictory"
ENTITY=None
RAW_DATA_AT = "contra_raw"
PROCESSED_DATA_AT = "contra_split"
SEED = 98765

DEFAULT_ARCH = "xlm-roberta-base"
MODEL_NAME = "contradictory-" + DEFAULT_ARCH

CLASSES = {0: "entailment", 1: "neutral", 2: "contradiction"}
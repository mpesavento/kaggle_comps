

# Constants
DATA_PATH = "data"
WANDB_PROJECT = "contradictory"
RAW_DATA_AT = "contra_raw"
PROCESSED_DATA_AT = "contra_split"

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {v:k for k,v in id2label.items()}

SEED = 98765

def seed_everything(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    
seed_everything(SEED)
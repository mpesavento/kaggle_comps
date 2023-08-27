import os
import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from datasets import Dataset, DatasetDict
import evaluate
from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaForSequenceClassification
from transformers import (
    TrainingArguments, Trainer, DataCollatorWithPadding)

from params import *

warnings.filterwarnings('ignore')

# load the HF accuracy fn
accuracy_fn = evaluate.load("accuracy")
# load the tokenizer into the global space; would be better to pass it around
tokenizer = XLMRobertaTokenizer.from_pretrained(DEFAULT_ARCH)


def download_data():
    """Grab dataset from artifact"""
    processed_data_at = wandb.use_artifact(f'{PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def get_df(processed_dataset_dir, is_test=False):
    """load a dataframe from the artifact data"""
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    if not is_test:
        # drop test for now, split in valid & train
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    else:
        df = df[df.Stage == 'test'].reset_index(drop=True)
    return df

def tokenize_function_batch(examples):
    tokenized_examples = tokenizer(examples["premise"], examples["hypothesis"], 
                                   truncation=True, padding=True, return_tensors="pt",)
    return tokenized_examples

def get_data(df):
    """
    Load the data from df into a dataset
    This is a bit more important if we are loading images/labels
    """
    train_dataset = Dataset.from_pandas(df[df["is_valid"]!=True])
    valid_dataset = Dataset.from_pandas(df[df["is_valid"]])
    datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})
    tokenized_datasets = datasets.map(tokenize_function_batch, batched=True)
    return tokenized_datasets

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # ent_ix = np.where(labels==label2id["entailment"])[0]
    # neut_ix = np.where(labels==label2id["neutral"])[0]
    # contra_ix = np.where(labels==label2id["contradiction"])[0]
    ent_ix = np.where(labels==0)[0]
    neut_ix = np.where(labels==1)[0]
    contra_ix = np.where(labels==2)[0]
    metrics = {
        "accuracy": accuracy_fn.compute(
            predictions=predictions, references=labels)["accuracy"],
        "acc_entailment": accuracy_fn.compute(
            predictions=predictions[ent_ix], references=labels[ent_ix])["accuracy"],
        "acc_neutral": accuracy_fn.compute(
            predictions=predictions[neut_ix], references=labels[neut_ix])["accuracy"],
        "acc_contradiction": accuracy_fn.compute(
            predictions=predictions[contra_ix], references=labels[contra_ix])["accuracy"],
    }
    return metrics

def get_predictions(dataset, trainer, metric_prefix=None):
    """Create predictions using the trainer object"""
    metric_prefix = metric_prefix or "validate"
    predictions = trainer.predict(dataset, metric_key_prefix=metric_prefix)
    X_pred = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    return X_pred, labels

def create_predictions_table(dataset, trainer):
    """Creates a wandb table with predictions and targets side by side"""
    X_pred, y_labels = get_predictions(dataset, trainer, metric_prefix="validate")
    if not np.array_equal(y_labels, [dataset[i]["label"] for i in range(len(dataset))]):
        raise ValueError("prediction labels do not match dataset labels")
    
    col_names = ["id", "premise", "hypothesis", "lang_abv", "label", "predict"]

    data_out = []
    for i, sample in tqdm(enumerate(dataset)):
        data_out.append({
            col:sample[col] for col in col_names[:-1]})
        data_out[-1][col_names[-1]] = X_pred[i]
    data_df = pd.DataFrame.from_records(data_out)

    # add the positive match column, True if matched target label, false otherwise
    data_df["is_correct"] = (data_df["label"]==data_df["predict"]).astype(int)
    
    table = wandb.Table(data=data_df)
    wandb.log({"val_pred_table":table})
    return table
    
def count_by_class(arr, cidxs): 
    return [(arr == n).sum(axis=(1,2)).numpy() for n in cidxs]

def log_hist(c):
    _, bins, _ = plt.hist(target_counts[c],  bins=10, alpha=0.5, density=True, label='target')
    _ = plt.hist(pred_counts[c], bins=bins, alpha=0.5, density=True, label='pred')
    plt.legend(loc='upper right')
    plt.title(CLASSES[c])
    img_path = f'figures/hist_val_{CLASSES[c]}'
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_path: wandb.Image(f'{img_path}.png', caption=img_path)})


# ---------------------------
# main

run = wandb.init(project=WANDB_PROJECT, entity=ENTITY, job_type="evaluation", tags=['staging'])

artifact = run.use_artifact('mpesavento/model-registry/contradictory_sentences:latest', type='model')

artifact_dir = Path(artifact.download())

_model_pth = artifact_dir.ls()[0]
model_path = _model_pth.parent.absolute()/_model_pth.stem

producer_run = artifact.logged_by()
wandb.config.update(producer_run.config)
config = wandb.config

processed_dataset_dir = download_data()
test_valid_df = get_df(processed_dataset_dir, is_test=True)
tokenized_datasets = get_data(test_valid_df)

# load the model

model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
output_dir = os.path.join(DATA_PATH, f"contradiction-training-{str(int(time.time()))}")

trainer_config = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_total_limit = 2,
    save_strategy="no",
    load_best_model_at_end=True,
    learning_rate=config.lr,
    num_train_epochs=config.num_epochs,
    weight_decay=0.01,
    logging_steps=1,
    report_to="wandb",  # enable logging to W&B
    run_name=MODEL_NAME,  # name of the W&B run (optional)
)

# set up the trainer
trainer = Trainer(
    model=model,
    args=trainer_config,
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,

)

val_metrics = learn.validate(ds_idx=1)
test_metrics = learn.validate(ds_idx=0)

val_metric_names = ['val_loss'] + [f'val_{x.name}' for x in learn.metrics]
val_results = {val_metric_names[i] : val_metrics[i] for i in range(len(val_metric_names))}
for k,v in val_results.items(): 
    wandb.summary[k] = v

test_metric_names = ['test_loss'] + [f'test_{x.name}' for x in learn.metrics]
test_results = {test_metric_names[i] : test_metrics[i] for i in range(len(test_metric_names))}
for k,v in test_results.items(): 
    wandb.summary[k] = v
    

val_table = create_predictions_table(test_valid_dls, trainer)

val_probs, val_targs = learn.get_preds(ds_idx=1)
val_preds = val_probs.argmax(dim=1)
class_idxs = CLASSES.keys()

target_counts = count_by_class(val_targs, class_idxs)
pred_counts = count_by_class(val_preds, class_idxs)

for c in class_idxs:
    log_hist(c)
    
val_count_df, val_disp = display_diagnostics(learner=learn, ds_idx=1, return_vals=True)
wandb.log({'val_confusion_matrix': val_disp.figure_})
val_ct_table = wandb.Table(dataframe=val_count_df)
wandb.log({'val_count_table': val_ct_table})

test_count_df, test_disp = display_diagnostics(learner=learn, ds_idx=0, return_vals=True)
wandb.log({'test_confusion_matrix': test_disp.figure_})
test_ct_table = wandb.Table(dataframe=test_count_df)
wandb.log({'test_count_table': test_ct_table})

run.finish()

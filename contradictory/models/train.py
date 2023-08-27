# imports
import argparse
import os
from pathlib import Path
import warnings
import time
from types import SimpleNamespace  # a wrapper around a datadict

import pandas as pd
import numpy as np
import kaggle
import wandb
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler
from transformers import XLMRobertaForSequenceClassification
from transformers import (
    TrainingArguments, Trainer, DataCollatorWithPadding,
    XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig)
from datasets import Dataset, DatasetDict
import evaluate

from params import *

warnings.filterwarnings('ignore')



device = "cpu"
if torch.cuda.is_available():
    print("Found GPU: ", torch.cuda.device_count())
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    mps_device = torch.device("mps")
    print("Found MPS, may not work on some torch ops!" )
    device = "mps"

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {v:k for k,v in id2label.items()}

default_config = SimpleNamespace(
    framework="torch",
    batch_size=16,
    num_epochs=5,
    lr=1e-5,
    arch=DEFAULT_ARCH,
    seed=SEED,
    log_preds=True,
    classifier_dropout=0.0,
    id2label=id2label,
    label2id=label2id,
)


# load the HF accuracy fn
accuracy_fn = evaluate.load("accuracy")
# load the tokenizer into the global space; would be better to pass it around
tokenizer = XLMRobertaTokenizer.from_pretrained(default_config.arch)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--num_epochs', type=int, default=default_config.num_epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config.arch, help='timm backbone architecture')
    argparser.add_argument('--seed', type=int, default=default_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=bool, default=default_config.log_preds, help='log model predictions')
    argparser.add_argument('--classifier_dropout', type=bool, default=default_config.classifier_dropout, help='pct linear dropout')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def seed_everything(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    
def download_data():
    processed_data_at = wandb.use_artifact(f'{PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def get_df(processed_dataset_dir, is_test=False):
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

def create_predictions_table(dataset, trainer, id2label):
    """Creates a wandb table with predictions and targets side by side"""
    predictions = trainer.predict(dataset, metric_key_prefix="validate")
    X_pred = np.argmax(predictions.predictions, axis=1)
    y_labels = predictions.label_ids
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
    wandb.log({"pred_table":table})
    return table

def log_final_metrics(trainer):
    scores = trainer.evaluate()
    for k,v in scores.items():
        wandb.summary[k] = v
        

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

def train(config):
    seed_everything(SEED)
    # init wandb
    run = wandb.init(project=WANDB_PROJECT, entity=None, job_type="training", config=config)

    processed_dataset_dir = download_data()
    df = get_df(processed_dataset_dir)
    tokenized_datasets = get_data(df)  # more space in this for hyperparameters

    config = wandb.config  # reload the instance config

    num_labels = len(np.unique(tokenized_datasets['train']["label"]))
    # fixed model arch for now
    xlm_roberta_config = XLMRobertaConfig.from_pretrained(config.arch)
    # set dropout prob
    # xlm_roberta_config.classifier_dropout = config.classifier_dropout

    # import pdb; pdb.set_trace()
    model = XLMRobertaForSequenceClassification.from_pretrained(config.arch, num_labels=num_labels)

    output_dir = os.path.join(DATA_PATH, f"contradiction-training-{str(int(time.time()))}")

    # log model on wandb
    # https://docs.wandb.ai/guides/integrations/huggingface#advanced-features
    os.environ["WANDB_LOG_MODEL"] = "end"
    
    trainer_config = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_total_limit = 2,
        save_strategy="epoch",
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
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,

    )
    
    # train it!
    model_trained = trainer.train()
    trainer.save_model()
    
    table = create_predictions_table(tokenized_datasets['validation'], trainer, id2label)
    wandb.finish()

if __name__ == '__main__':
    parse_args()
    train(default_config)
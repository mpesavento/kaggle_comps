import os
import time
from pathlib import Path
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display, Markdown

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
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    else:
        df = df[df.Stage != 'train'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
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
    # when passed to datablock, this will return test at index 0 and valid at index 1
    
    valid_dataset = Dataset.from_pandas(df[df["is_valid"]])
    if "train" in df.Stage.unique():
        train_dataset = Dataset.from_pandas(df[df["is_valid"]!=True])
        datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})
    elif "test" in df.Stage.unique():
        test_dataset = Dataset.from_pandas(df[df["is_valid"]!=True])
        datasets = DatasetDict({"test": test_dataset, "validation": valid_dataset})
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
    """
    Create predictions using the trainer object
    Returns tuple: (probabilities, predictions, targets)
    """
    metric_prefix = metric_prefix or "validate"
    predictions = trainer.predict(dataset, metric_key_prefix=metric_prefix)
    X_pred = np.argmax(predictions.predictions, axis=1)
    targets = predictions.label_ids
    return predictions.predictions, X_pred, targets

def create_predictions_table(dataset, trainer):
    """Creates a wandb table with predictions and targets side by side"""
    X_proba, X_pred, targets = get_predictions(dataset, trainer, metric_prefix="validate")
    if not np.array_equal(targets, [dataset[i]["label"] for i in range(len(dataset))]):
        raise ValueError("prediction targets do not match dataset labels")
    
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
    return [(arr == n).sum() for n in cidxs]

def log_label_hist(target_counts, pred_counts):
    """Get a histogram of target/pred counts"""
    plt.bar(CLASSES.values(), target_counts, alpha=0.5, label="target")
    plt.bar(CLASSES.values(), pred_counts, alpha=0.5, label="pred")
    plt.legend(loc='upper right')
    plt.title("target/pred by class")
    img_folder = "figures"
    os.makedirs(img_folder, exist_ok=True)
    img_name = f"hist_val_targetpred"
    img_path = os.path.join(img_folder, img_name)
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_name: wandb.Image(f'{img_path}.png', caption=img_name)})

def log_lang_hist(valid_df, clsix):
    """Create histograms of the target/pred over each language"""
    target_lang_counts = valid_df[valid_df.label == clsix].groupby("lang_abv")["label"].count()
    pred_lang_counts = valid_df[valid_df.predict == clsix].groupby("lang_abv")["predict"].count()
    plt.bar(langs, target_lang_counts, alpha=0.5, label="target")
    plt.bar(langs, pred_lang_counts, alpha=0.5, label="pred")
    plt.legend(loc='upper right')
    plt.title(f"{CLASSES[clsix]}: target/pred by lang")
    img_folder = "figures"
    os.makedirs(img_folder, exist_ok=True)
    img_name = f"hist_val_lang_{CLASSES[clsix]}"
    img_path = os.path.join(img_folder, img_name)
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_name: wandb.Image(f'{img_path}.png', caption=img_name)})
    

def get_lang_accuracy(valid_df):
    """Do some fancy categorical plots over languages"""
    img_folder = "figures"
    os.makedirs(img_folder, exist_ok=True)
    sns.catplot(data=valid_df, x="label_name", y="is_correct", hue="lang_abv", kind="bar",
                height=8, aspect=15/8)
    plt.title("accuracy by label")
    img_name = f"val_acc_label"
    img_path = os.path.join(img_folder, img_name)
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_name: wandb.Image(f'{img_path}.png', caption=img_name)})
    
    sns.catplot(data=valid_df, x="lang_abv", y="is_correct", hue="label_name", kind="bar",
                height=8, aspect=15/8)
    plt.title("accuracy by language")
    img_name = f"val_acc_lang"
    img_path = os.path.join(img_folder, img_name)
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_name: wandb.Image(f'{img_path}.png', caption=img_name)})
    

def display_diagnostics(trainer, dataset, metric_prefix=None, return_vals=False):
    """
    Display a confusion matrix for the trainer.
    
    You can create a test dataloader using the `test_dl()` method like so:
    >> dls = ... # You usually create this from the DataBlocks api, in this library it is get_data()
    >> tdls = dls.test_dl(test_dataframe, with_labels=True)
    
    See: https://docs.fast.ai/tutorial.pets.html#adding-a-test-dataloader-for-inference
    
    """
    metric_prefix = metric_prefix or "validate"
    y_proba, y_pred, targets = get_predictions(dataset, trainer, metric_prefix=metric_prefix)
    classes = list(CLASSES.values())
    y_true = targets
    
    tdf, pdf = [pd.DataFrame(r).value_counts().to_frame(c) for r,c in zip((y_true, y_pred) , ['y_true', 'y_pred'])]
    countdf = tdf.join(pdf, how='outer').reset_index(drop=True).fillna(0).astype(int).rename(index=CLASSES)
    countdf = countdf/countdf.sum() 
    display(Markdown('### % Of samples In Each Class'))
    display(countdf.style.format('{:.1%}'))
    
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred,
                                                   display_labels=classes,
                                                   normalize='pred')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(10) 
    disp.ax_.set_title('Confusion Matrix', fontdict={'fontsize': 32, 'fontweight': 'medium'})
    fig.show()
    fig.autofmt_xdate(rotation=45)

    if return_vals: return countdf, disp

# ---------------------------
# main

run = wandb.init(project=WANDB_PROJECT, entity=ENTITY, job_type="evaluation", tags=['staging'])

artifact = run.use_artifact('mpesavento/model-registry/contradictory_sentences:latest', type='model')

artifact_dir = Path(artifact.download())

_model_pth = artifact_dir #.ls()[0]
model_path = _model_pth.parent.absolute()/_model_pth.stem

producer_run = artifact.logged_by()
wandb.config.update(producer_run.config)
config = wandb.config

processed_dataset_dir = download_data()
test_valid_df = get_df(processed_dataset_dir, is_test=True)
tokenized_datasets = get_data(test_valid_df)

# load the model

model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
output_dir = os.path.join(DATA_PATH, f"contradiction-eval-{str(int(time.time()))}")

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
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,

)

val_metrics = trainer.evaluate(tokenized_datasets['validation'], metric_key_prefix="val")
test_metrics = trainer.evaluate(tokenized_datasets['test'], metric_key_prefix="test")
for k,v in val_metrics.items(): 
    wandb.summary[k] = v
for k,v in test_metrics.items(): 
    wandb.summary[k] = v

# log the validation results
val_table = create_predictions_table(tokenized_datasets['validation'], trainer)

# make the distribution histograms
val_probs, val_preds, val_targs = get_predictions(tokenized_datasets["validation"], trainer)
class_idxs = CLASSES.keys()

# create plots for label and language accuracy
valid_df = test_valid_df[test_valid_df["is_valid"]].copy()
valid_df["predict"] = val_preds
valid_df["label_name"] = valid_df["label"].map(lambda x: CLASSES[x])
valid_df["is_correct"] = (valid_df["label"]==valid_df["predict"]).astype(int)

get_lang_accuracy(valid_df)

langs = valid_df.lang_abv.unique()
for c in class_idxs:
    log_lang_hist(valid_df, c)

# not as important with categorical classification, more important with segmentation
target_counts = count_by_class(val_targs, class_idxs)
pred_counts = count_by_class(val_preds, class_idxs)
log_label_hist(target_counts, pred_counts)


val_count_df, val_disp = display_diagnostics(trainer, tokenized_datasets['validation'], return_vals=True)
wandb.log({'val_confusion_matrix': val_disp.figure_})
val_ct_table = wandb.Table(dataframe=val_count_df)
wandb.log({'val_count_table': val_ct_table})

test_count_df, test_disp = display_diagnostics(trainer, tokenized_datasets['test'], return_vals=True)
wandb.log({'test_confusion_matrix': test_disp.figure_})
test_ct_table = wandb.Table(dataframe=test_count_df)
wandb.log({'test_count_table': test_ct_table})

run.finish()

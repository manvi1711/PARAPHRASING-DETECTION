#Install required libraries
!pip install transformers datasets torch scikit-learn

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

#Disable Weights & Biases (W&B) logging
os.environ["wNDS DISABLED"]="true"

#Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device )

#Load MRPC dataset from Hugging Face
dataset = load_dataset("glue", "mrpc")

#Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#Install required libraries
!pip install transformers datasets torch scikit-learn

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

#Disable Weights & Biases (W&B) logging
os.environ["WANDB_DISABLED"]="true"

#Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device )

#Load MRPC dataset from Hugging Face
dataset = load_dataset("glue", "mrpc")

#Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#Tokenization function
def tokenize_function(example):
  return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=128)

#Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

#Remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets.set_format("torch")

#Split datasets
train_dataset = tokenized_datasets["train"]
valid_dataset = tokenized_datasets["validation"]

#Load BERT model for binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device) #Move model to GPU

# Define evaluation metrics
def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=1)
  acc = accuracy_score(labels, predictions)
  f1 = f1_score(labels, predictions)
  return {"accuracy": acc, "f1_score": f1}

#Training arguments (Fixed eval_strategy)
training_args = TrainingArguments(
  output_dir="./results",
  eval_strategy="epoch", #Fixed
  save_strategy="epoch",
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
  logging_dir="./logs",
  logging_steps=500,
  load_best_model_at_end=True,
 )

#Trainer API (Removed tokenizer=tokenizer)
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=valid_dataset,
  compute_metrics=compute_metrics,
)

#Train model
trainer.train()

#Evaluate the model
results = trainer.evaluate()
print(results)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

#Remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns(["sentencel", "sentence2", "idx"])
tokenized_datasets.set_format("torch")

#Split datasets
train_dataset = tokenized_datasets["train"]
valid_dataset = tokenized_datasets["validation"]

#Load BERT model for binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels-2)
model.to(device) #Move model to GPU

# Define evaluation metrics
def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis-1)
  acc = accuracy_score(labels, predictions)
  f1 = f1_score(labels, predictions)
  return {"accuracy": acc, "f1_score": f1}

#Training arguments (Fixed eval_strategy)
training_args = TrainingArguments(
  output_dir="./results",
  eval_strategy="epoch", #Fixed
  save_strategy="epoch",
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
  logging_dir="./logs",
  logging_steps=500,
  load_best_model_at_end=True,
 )

#Trainer API (Removed tokenizer=tokenizer)
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=valid_dataset,
  compute_metrics=compute_metrics,
)

#Train model
trainer.train()

#Evaluate the model
results = trainer.evaluate()
print(results)

from transformers import  BertTokenizer, BertForSequenceClassification
import torch

#load pretrained BERT model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_lables = 2)

#example sentence
sentence1 = "The sky is clear today."
sentence2 = "The weather is rainy today."

#tozenizer
inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)

#model
with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits
  predictions = torch.argmax(logits, dim=1)

#output
print("Paraphrase" if predictions.item() == 1 else "Not Paraphrase")
import argparse

import sys
sys.path.append("..")

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from inflation import ECDSep
import evaluate
import random
import numpy as np

def compute_accuracy(eval_dataloader, dataset_name):
    metric = evaluate.load("glue", dataset_name)
    for batch in eval_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
        output = model(**batch)
      
      logits = output.logits
      if dataset_name != "stsb":
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
      else:
        metric.add_batch(predictions=logits.flatten(), references=batch["labels"])

    results = metric.compute()
    return results
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--optimizer', default='ECDSep', type=str, help="name of the optimizer among adam, adamw, sgd and ECDSep")
  parser.add_argument('--lr', default=0.04, type=float, help="learning rate value")
  parser.add_argument('--momentum', default=0.99, type=float, help="momentum value (for sgd)")
  parser.add_argument('--nu', default=1e-5, type=float, help="nu value (for ECDSep)")
  parser.add_argument('--eta', default=1.4, type=float, help="eta value (for ECDSep)")
  parser.add_argument('--consEn', default=True, type=bool, help="whether the energy conservation is true or false (for ECDSep)")
  parser.add_argument('--F0', default=0., type=float, help="F0 value (for ECDSep)")
  parser.add_argument('--deltaEn', default=0.0, type=float, help="deltaEn value (for ECDSep)")
  parser.add_argument('--s', default=1., type=float, help="s value (for ECDSep)")
  parser.add_argument('--epochs', default=3, type=int, help="number of epochs")
  parser.add_argument('--dataset', default="all", type=str, help="name of the glue dataset")
  parser.add_argument('--seed', default=42, type=int, help="random seed value")
  parser.add_argument('--wd', default=0., type=float, help="weight decay value")

  args = parser.parse_args()
  opt = args.optimizer
  lr = args.lr
  nu = args.nu
  eta = args.eta
  consEn = args.consEn
  s = args.s
  deltaEn = args.deltaEn
  momentum = args.momentum
  F0 = args.F0
  epochs = args.epochs
  datasets_name = args.dataset
  seed = args.seed
  wd = args.wd

  torch.manual_seed(seed)
  random.seed(seed)
  torch.cuda.manual_seed(seed)

  if datasets_name == 'all':
    datasets = ["mrpc", "qqp", "cola", "qnli", "rte", "sst2", "mnli", "stsb"]
  else:
    datasets = [datasets_name]

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print("Device:", device)

  best_losses, best_accuracies = [], []
  for dataset_name in datasets:
    print('Dataset:', dataset_name)

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if dataset_name == "mrpc":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

      tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    elif dataset_name == "qqp":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
        return tokenizer(example["question1"], example["question2"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

      tokenized_dataset = tokenized_dataset.remove_columns(["question1", "question2", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    elif dataset_name == "cola":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
          return tokenizer(example["sentence"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

      tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    elif dataset_name == "mnli":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
          return tokenizer(example["premise"], example["hypothesis"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

      tokenized_dataset = tokenized_dataset.remove_columns(["premise", "hypothesis", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    elif dataset_name == "qnli":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
          return tokenizer(example["question"], example["sentence"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

      tokenized_dataset = tokenized_dataset.remove_columns(["question", "sentence", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    elif dataset_name == "rte":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
          return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

      tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    elif dataset_name == "sst2":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
          return tokenizer(example["sentence"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

      tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    elif dataset_name == "stsb":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
          return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

      tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    elif dataset_name == "wnli":
      raw_dataset = load_dataset("glue", dataset_name)

      def tokenize_function(example):
          return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

      tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
      
      tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
      tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
      tokenized_dataset.set_format("torch")
      tokenized_dataset["train"].column_names

    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    if dataset_name == "mnli":
        eval_dataloader_m = DataLoader(tokenized_dataset["validation_matched"], batch_size=8, collate_fn=data_collator)
        eval_dataloader_mm = DataLoader(tokenized_dataset["validation_mismatched"], batch_size=8, collate_fn=data_collator)
    else:
        eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=8, collate_fn=data_collator)

    if dataset_name == "stsb":
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
        model.config.problem_type = "regression"
    elif dataset_name == "mnli":
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    model.to(device)

    if opt == 'adamw':
      optimizer = AdamW(model.parameters(), lr=lr, weight_decay = wd)

    elif opt == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = wd)

    elif opt == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = wd, momentum=momentum)

    elif opt == 'ECDSep':
      optimizer = ECDSep(model.parameters(), lr=lr, eta=eta, nu=nu, consEn=consEn, deltaEn=deltaEn, F0=F0, s=s, weight_decay=wd)
      
    num_epochs = epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    print('Num training steps:', num_training_steps)

    progress_bar = tqdm(range(num_training_steps))

    t = 0
    for epoch in range(num_epochs):
      model.train()
      for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        def closure():
          return loss
        optimizer.step(closure)
        if opt in ["adam", "adamw"]:
          lr_scheduler.step()
        progress_bar.update()
        
        if t == 0:
          best_loss = loss.item()
        else:
          if loss.item() < best_loss:
            best_loss = loss.item()
        t += 1

      with torch.no_grad():
        model.eval()
        if dataset_name == "cola":
          met = 'matthews_correlation'
        elif dataset_name == "stsb":
          met = "spearmanr"
        elif dataset_name in ["mrpc", "qqp"]:
          met = "f1"
        else: 
          met = "accuracy"
        if dataset_name != "mnli":
            results = compute_accuracy(eval_dataloader, dataset_name)
            print(results)
            test_accuracy = results[met]
            print(met+'=', test_accuracy)
            if epoch == 0:
                best_test_accuracy = test_accuracy
            else:
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
        else:
            results_m = compute_accuracy(eval_dataloader_m, dataset_name)
            results_mm = compute_accuracy(eval_dataloader_mm, dataset_name)
            print(results_m, results_mm)
            test_accuracy_m = results_m[met]
            test_accuracy_mm = results_mm[met]
            print(met+' m=', test_accuracy_m,', '+met+' mm=', test_accuracy_mm)
            test_accuracy = (test_accuracy_m + test_accuracy_mm) / 2
            if epoch == 0:
                best_test_accuracy = test_accuracy
            else:
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy

    best_losses.append(best_loss)
    best_accuracies.append(best_test_accuracy)
    print("Best "+met+" for "+opt+" on "+dataset_name+" is ", best_test_accuracy)
    print("Minimum loss for "+opt+" on "+dataset_name+" is ", best_loss)


  if datasets_name == "all":
    avg_best_accuracy = np.mean(best_accuracies)
    print("Average best accuracy for "+opt+" on "+dataset_name+" is ", np.mean(avg_best_accuracy))

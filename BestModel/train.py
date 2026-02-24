import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from dont_patronize_me import DontPatronizeMe

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "microsoft/deberta-v3-base"
MAX_LENGTH     = 128
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 1e-5
WARMUP_RATIO   = 0.1
WEIGHT_DECAY   = 0.01
FOCAL_GAMMA    = 2.0   # focal loss focusing parameter
LABEL_SMOOTHING = 0.1
FGM_EPSILON    = 0.5   # adversarial perturbation magnitude
DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR     = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    Down-weights easy examples so the model focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha          = alpha          # class weights tensor [w0, w1]
        self.gamma          = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        n_classes = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)

        # Get p_t for each sample (prob of the true class)
        p_t       = (probs * F.one_hot(targets, n_classes)).sum(dim=1)
        alpha_t   = self.alpha[targets]
        focal_w   = alpha_t * (1 - p_t) ** self.gamma

        # Cross entropy with smooth targets
        ce_loss   = -(smooth_targets * log_probs).sum(dim=1)
        loss      = focal_w * ce_loss
        return loss.mean()


# ── FGM Adversarial Training ──────────────────────────────────────────────────
class FGM:
    """
    Fast Gradient Method adversarial training.
    Perturbs the word embeddings during training to improve robustness.
    Used by the 1st-place PALI-NLP team at SemEval-2022 Task 4.
    """
    def __init__(self, model, epsilon=0.5):
        self.model   = model
        self.epsilon = epsilon
        self.backup  = {}

    def attack(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data():
    dpm = DontPatronizeMe(DATA_DIR, os.path.join(DATA_DIR, 'task4_test.tsv'))
    dpm.load_task1()
    dpm.load_test()

    full_df = dpm.train_task1_df  # par_id, art_id, keyword, country, text, label, orig_label

    train_ids = pd.read_csv(
        os.path.join(DATA_DIR, 'train_semeval_parids-labels.csv')
    )
    dev_ids = pd.read_csv(
        os.path.join(DATA_DIR, 'dev_semeval_parids-labels.csv')
    )

    train_ids['par_id'] = train_ids['par_id'].astype(str).str.strip()
    dev_ids['par_id']   = dev_ids['par_id'].astype(str).str.strip()
    full_df['par_id']   = full_df['par_id'].astype(str).str.strip()

    train_df = full_df[full_df['par_id'].isin(train_ids['par_id'])].reset_index(drop=True)
    dev_df   = full_df[full_df['par_id'].isin(dev_ids['par_id'])].reset_index(drop=True)
    test_df  = dpm.test_set_df

    print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
    print(f"Train PCL: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    print(f"Dev   PCL: {dev_df['label'].sum()}   ({dev_df['label'].mean()*100:.1f}%)")

    return train_df, dev_df, test_df


# ── Dataset ───────────────────────────────────────────────────────────────────
class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


class PCLTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ── Class Weights ─────────────────────────────────────────────────────────────
def compute_class_weights(labels):
    counts  = np.bincount(labels)
    total   = len(labels)
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float).to(device)


# ── Train One Epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimiser, scheduler, loss_fn, fgm):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc='Training'):
        optimiser.zero_grad()
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()

        # FGM adversarial pass
        fgm.attack()
        adv_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        adv_loss    = loss_fn(adv_outputs.logits, labels)
        adv_loss.backward()
        fgm.restore()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds = (np.array(all_probs) >= threshold).astype(int)
    f1    = f1_score(all_labels, preds, pos_label=1)
    print(classification_report(all_labels, preds, target_names=['No PCL', 'PCL']))
    return f1, np.array(all_probs), np.array(all_labels)


# ── Threshold Tuning ──────────────────────────────────────────────────────────
def tune_threshold(probs, labels):
    best_t  = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    print(f"Best threshold: {best_t:.2f} | Best dev F1: {best_f1:.4f}")
    return best_t, best_f1


# ── Predict on Test ───────────────────────────────────────────────────────────
def predict_test(model, loader, threshold):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting test'):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = torch.softmax(outputs.logits, dim=-1)[:, 1]
            preds          = (probs.cpu().numpy() >= threshold).astype(int)
            all_preds.extend(preds)
    return all_preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_df, dev_df, test_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = PCLDataset(train_df['text'], train_df['label'], tokenizer, MAX_LENGTH)
    dev_dataset   = PCLDataset(dev_df['text'],   dev_df['label'],   tokenizer, MAX_LENGTH)
    test_dataset  = PCLTestDataset(test_df['text'], tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    class_weights = compute_class_weights(train_df['label'].values)
    loss_fn       = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    fgm           = FGM(model, epsilon=FGM_EPSILON)

    optimiser     = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_cosine_schedule_with_warmup(optimiser, warmup_steps, total_steps)

    best_f1        = 0.0
    best_threshold = 0.5

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        train_loss = train_epoch(model, train_loader, optimiser, scheduler, loss_fn, fgm)
        print(f"Train loss: {train_loss:.4f}")

        f1, dev_probs, dev_labels = evaluate(model, dev_loader, threshold=0.5)
        print(f"Dev F1 (threshold=0.5): {f1:.4f}")

        t, tuned_f1 = tune_threshold(dev_probs, dev_labels)

        if tuned_f1 > best_f1:
            best_f1        = tuned_f1
            best_threshold = t
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
            print(f"Saved best model — F1: {best_f1:.4f} at threshold {best_threshold:.2f}")

    print(f"\nFinal best dev F1: {best_f1:.4f} at threshold {best_threshold:.2f}")

    # Load best model and generate final predictions
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt')))

    # Dev predictions
    _, dev_probs, dev_labels = evaluate(model, dev_loader, threshold=best_threshold)
    dev_preds = (dev_probs >= best_threshold).astype(int)
    pred_dir  = os.path.join(os.path.dirname(__file__), '..', 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    with open(os.path.join(pred_dir, 'dev.txt'), 'w') as f:
        for p in dev_preds:
            f.write(f"{p}\n")
    print(f"Saved dev.txt ({len(dev_preds)} predictions)")

    # Test predictions
    test_preds = predict_test(model, test_loader, best_threshold)
    with open(os.path.join(pred_dir, 'test.txt'), 'w') as f:
        for p in test_preds:
            f.write(f"{p}\n")
    print(f"Saved test.txt ({len(test_preds)} predictions)")


if __name__ == '__main__':
    main()
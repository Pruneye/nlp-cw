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
MODEL_NAME      = "microsoft/deberta-v3-large"
MAX_LENGTH      = 128
BATCH_SIZE      = 8
GRAD_ACCUM      = 2
EPOCHS          = 10
LR              = 8e-6
WARMUP_RATIO    = 0.1
WEIGHT_DECAY    = 0.01
FOCAL_GAMMA     = 2.0
LABEL_SMOOTHING = 0.1
FGM_EPSILON     = 0.5
DATA_DIR        = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR      = os.path.dirname(__file__)
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, 'checkpoints')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
METRICS_PATH = os.path.join(OUTPUT_DIR, 'training_metrics.csv')

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
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)
        p_t       = (probs * F.one_hot(targets, n_classes)).sum(dim=1)
        alpha_t   = self.alpha[targets]
        focal_w   = alpha_t * (1 - p_t) ** self.gamma
        ce_loss   = -(smooth_targets * log_probs).sum(dim=1)
        return (focal_w * ce_loss).mean()


# ── FGM Adversarial Training ──────────────────────────────────────────────────
class FGM:
    """
    Fast Gradient Method adversarial training.
    Perturbs word embeddings to improve robustness.
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

    full_df = dpm.train_task1_df

    train_ids = pd.read_csv(os.path.join(DATA_DIR, 'train_semeval_parids-labels.csv'))
    dev_ids   = pd.read_csv(os.path.join(DATA_DIR, 'dev_semeval_parids-labels.csv'))

    train_ids['par_id'] = train_ids['par_id'].astype(str).str.strip()
    dev_ids['par_id']   = dev_ids['par_id'].astype(str).str.strip()
    full_df['par_id']   = full_df['par_id'].astype(str).str.strip()

    train_df = full_df[full_df['par_id'].isin(train_ids['par_id'])].reset_index(drop=True)
    # Preserve dev order to match predictions row-by-row
    dev_df   = dev_ids[['par_id']].merge(full_df, on='par_id', how='left').reset_index(drop=True)
    test_df  = dpm.test_set_df

    print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
    print(f"Train PCL: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    print(f"Dev   PCL: {dev_df['label'].sum()}   ({dev_df['label'].mean()*100:.1f}%)")

    return train_df, dev_df, test_df


# ── Dataset ───────────────────────────────────────────────────────────────────
class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt'
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
            list(texts), truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt'
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


# ── Checkpoint Helpers ────────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimiser, scheduler, scaler, best_f1, best_threshold):
    path = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimiser_state': optimiser.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state':    scaler.state_dict(),
        'best_f1':         best_f1,
        'best_threshold':  best_threshold,
    }, path)
    print(f"Checkpoint saved: {path}")


def find_latest_checkpoint():
    path = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    if not os.path.exists(path):
        return None, 0
    ckpt = torch.load(path, map_location='cpu')
    return path, ckpt['epoch']


# ── Train One Epoch (fp16) ────────────────────────────────────────────────────
def train_epoch(model, loader, optimiser, scheduler, loss_fn, fgm, scaler):
    model.train()
    total_loss = 0
    optimiser.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc='Training')):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        # Normal forward + backward (fp16)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
        scaler.scale(loss).backward()

        # FGM adversarial pass (fp16)
        fgm.attack()
        with torch.cuda.amp.autocast():
            adv_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            adv_loss    = loss_fn(adv_outputs.logits, labels)
        scaler.scale(adv_loss).backward()
        fgm.restore()

        if (i + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            optimiser.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM

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

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
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
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            preds = (probs.cpu().numpy() >= threshold).astype(int)
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
    scaler = torch.cuda.amp.GradScaler()

    class_weights = compute_class_weights(train_df['label'].values)
    loss_fn       = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    fgm           = FGM(model, epsilon=FGM_EPSILON)

    optimiser    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(optimiser, warmup_steps, total_steps)

    best_f1        = 0.0
    best_threshold = 0.5
    start_epoch    = 0

    # Resume from checkpoint if one exists
    ckpt_path, last_epoch = find_latest_checkpoint()
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimiser.load_state_dict(ckpt['optimiser_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        best_f1        = ckpt['best_f1']
        best_threshold = ckpt['best_threshold']
        start_epoch    = ckpt['epoch']
        print(f"Resumed at epoch {start_epoch} | best F1 so far: {best_f1:.4f}")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        train_loss = train_epoch(model, train_loader, optimiser, scheduler, loss_fn, fgm, scaler)
        print(f"Train loss: {train_loss:.4f}")

        f1, dev_probs, dev_labels = evaluate(model, dev_loader, threshold=0.5)
        print(f"Dev F1 (threshold=0.5): {f1:.4f}")

        t, tuned_f1 = tune_threshold(dev_probs, dev_labels)

        if tuned_f1 > best_f1:
            best_f1        = tuned_f1
            best_threshold = t
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
            print(f"Saved best model — F1: {best_f1:.4f} at threshold {best_threshold:.2f}")

        # Log metrics
        metrics_row = {
            'epoch':         epoch + 1,
            'train_loss':    train_loss,
            'dev_f1_05':     f1,
            'best_tuned_f1': tuned_f1,
            'threshold':     t,
        }
        metrics_df   = pd.DataFrame([metrics_row])
        write_header = not os.path.exists(METRICS_PATH)
        metrics_df.to_csv(METRICS_PATH, mode='a', header=write_header, index=False)

        # Save single rolling checkpoint
        save_checkpoint(epoch + 1, model, optimiser, scheduler, scaler, best_f1, best_threshold)

    print(f"\nFinal best dev F1: {best_f1:.4f} at threshold {best_threshold:.2f}")

    # Load best model and generate final predictions
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
    if not os.path.exists(best_model_path):
        print("Warning: best_model.pt not found, saving current model state")
        torch.save(model.state_dict(), best_model_path)
    model.load_state_dict(torch.load(best_model_path))

    pred_dir = os.path.join(os.path.dirname(__file__), '..', 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    # Dev predictions + raw probs for PR curve
    _, dev_probs, dev_labels = evaluate(model, dev_loader, threshold=best_threshold)
    dev_preds = (dev_probs >= best_threshold).astype(int)
    np.save(os.path.join(pred_dir, 'dev_probs.npy'), dev_probs)

    with open(os.path.join(pred_dir, 'dev.txt'), 'w') as f:
        for p in dev_preds:
            f.write(f"{p}\n")
    print(f"Saved dev.txt ({len(dev_preds)} predictions)")
    print(f"Saved dev_probs.npy ({len(dev_probs)} probabilities)")

    # Test predictions
    test_preds = predict_test(model, test_loader, best_threshold)
    with open(os.path.join(pred_dir, 'test.txt'), 'w') as f:
        for p in test_preds:
            f.write(f"{p}\n")
    print(f"Saved test.txt ({len(test_preds)} predictions)")


if __name__ == '__main__':
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt

FEATURE_CSV = "../Bias_Detector/data/processed/bert_features.csv"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32
LR          = 3e-4
WD          = 1e-4
EPOCHS      = 30
PATIENCE    = 3
HIDDEN_DIMS = [1024, 512, 256]
MAX_GRAD    = 1.0
MODEL_PATH  = "best_weights.pth"

df = pd.read_csv(FEATURE_CSV)
X = df[[f"feat_{i}" for i in range(768)]].values.astype(np.float32)
y = df["bias_label"].str.lower().values

le     = LabelEncoder()
y_enc  = le.fit_transform(y)
labels = list(le.classes_)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=None, num_classes=3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims or []:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.5),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLPClassifier(input_dim=768, hidden_dims=HIDDEN_DIMS, num_classes=len(labels)).to(DEVICE)

class_w = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
weight_tensor = torch.tensor(class_w, dtype=torch.float32).to(DEVICE)
criterion     = nn.CrossEntropyLoss(weight=weight_tensor)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
total_steps = len(train_loader) * EPOCHS
scheduler   = OneCycleLR(
    optimizer, max_lr=LR,
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=1e4
)

best_val_loss = float("inf")
no_improve    = 0

train_losses = []
val_losses   = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    all_preds, all_trues = [], []
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * Xb.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_trues.extend(yb.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_acc  = accuracy_score(all_trues, all_preds)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    v_preds, v_trues = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            loss   = criterion(logits, yb)

            val_loss += loss.item() * Xb.size(0)
            v_preds.extend(logits.argmax(dim=1).cpu().numpy())
            v_trues.extend(yb.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_acc  = accuracy_score(v_trues, v_preds)
    val_losses.append(val_loss)

    print(f"Epoch {epoch:02d}: "
          f"Train Loss={train_loss:.4f}, Acc={train_acc:.3f} | "
          f"Val Loss={val_loss:.4f}, Acc={val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"No improvement in {PATIENCE} epochs—stopping early.")
            break

plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)   + 1), val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

def evaluate(loader, name):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            preds  = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)

    print(f"\n--- {name} Results ---")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=labels,
        cmap="Blues"
    )
    plt.title(f"{name} Confusion Matrix")
    plt.show()

model.load_state_dict(torch.load(MODEL_PATH))
evaluate(train_loader, "TRAIN")
evaluate(val_loader,   "VAL")
evaluate(test_loader,  "TEST")

with torch.no_grad():
    all_X = torch.from_numpy(X).to(DEVICE)
    logits = model(all_X)
    all_preds = logits.argmax(dim=1).cpu().numpy()

all_pred_labels = le.inverse_transform(all_preds)

full_results = pd.DataFrame({
    'source': df['source'],         
    'predicted': all_pred_labels
})

counts = (
    full_results
    .groupby(['source', 'predicted'])
    .size()
    .unstack(fill_value=0)
)

counts['top_bias'] = counts.idxmax(axis=1)

print("\nPrediction counts per source/bias:")
print(counts)

# counts.to_csv("source_bias_counts_with_top.csv")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

CSV_PATH     = "./augmented_bert_features.csv"
FEATURE_COLS = [f"aug_feat_{i}" for i in range(768)]
BATCH_SIZE   = 32
LR           = 6.069421337601868e-05
WD           = 3.2330898413333895e-06
EPOCHS       = 50
PATIENCE     = 10
HIDDEN_DIMS  = [651]
DROPOUT_RATE = 0.3447795324043547
LAMBDA_GRL   = 0.5   
SEED         = 42

OUTPUT_DIR = "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Using device: {device}")

df = pd.read_csv(CSV_PATH)
SOURCE2ID = {s:i for i,s in enumerate(sorted(df['source'].unique()))}
BIAS2ID   = {b:i for i,b in enumerate(sorted(df['bias_label'].unique()))}
ID2BIAS   = {i:b for b,i in BIAS2ID.items()}
ID2SOURCE = {i:s for s,i in SOURCE2ID.items()}

df_rest, df_test = train_test_split(df, test_size=0.2, stratify=df['bias_label'], random_state=SEED)
df_train, df_val = train_test_split(df_rest, test_size=0.25, stratify=df_rest['bias_label'], random_state=SEED)

def make_ds(split):
    X = torch.tensor(split[FEATURE_COLS].values, dtype=torch.float32)
    yb = torch.tensor(split['bias_label'].map(BIAS2ID).values, dtype=torch.long)
    ys = torch.tensor(split['source'].map(SOURCE2ID).values, dtype=torch.long)
    return TensorDataset(X, yb, ys)

train_ds = make_ds(df_train)
val_ds   = make_ds(df_val)
test_ds  = make_ds(df_test)
loaders = {
    'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
    'val':   DataLoader(val_ds,   batch_size=BATCH_SIZE),
    'test':  DataLoader(test_ds,  batch_size=BATCH_SIZE),
}


from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lamb, None

def grl(x, lamb):
    return GradReverse.apply(x, lamb)


class DualHeadMLP(nn.Module):
    def __init__(self, inp_dim, hidden_dims, drop, n_bias, n_src):
        super().__init__()
        layers, prev = [], inp_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)
            ]
            prev = h
        self.shared = nn.Sequential(*layers)
        self.head_bias   = nn.Linear(prev, n_bias)
        self.head_source = nn.Linear(prev, n_src)

    def forward(self, x, grl_lambda=0.0):
        features = self.shared(x)
        out_bias   = self.head_bias(features)
        rev_feat   = grl(features, grl_lambda)
        out_source = self.head_source(rev_feat)
        return out_bias, out_source

model = DualHeadMLP(len(FEATURE_COLS), HIDDEN_DIMS, DROPOUT_RATE, len(BIAS2ID), len(SOURCE2ID)).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
train_bias_losses, train_src_losses, val_bias_losses = [], [], []
best_val_loss = float('inf')
patience_cnt  = 0
for epoch in range(1, EPOCHS+1):
    model.train()
    sum_b, sum_s, n = 0.0, 0.0, 0
    for Xb, yb, ys in loaders['train']:
        Xb, yb, ys = Xb.to(device), yb.to(device), ys.to(device)
        optimizer.zero_grad()
        ob, os_ = model(Xb, grl_lambda=LAMBDA_GRL)
        lb = criterion(ob, yb)
        ls = criterion(os_, ys)
        loss = lb + ls
        loss.backward()
        optimizer.step()
        bs = Xb.size(0)
        sum_b += lb.item() * bs
        sum_s += ls.item() * bs
        n += bs
    train_bias = sum_b / n
    train_src  = sum_s / n
    train_bias_losses.append(train_bias)
    train_src_losses.append(train_src)
    model.eval()
    v_sum, v_n = 0.0, 0
    with torch.no_grad():
        for Xv, yv, _ in loaders['val']:
            Xv, yv = Xv.to(device), yv.to(device)
            vb, _ = model(Xv)
            v_sum += criterion(vb, yv).item() * Xv.size(0)
            v_n += Xv.size(0)
    val_bias = v_sum / v_n
    val_bias_losses.append(val_bias)
    print(f"Epoch {epoch:2d} | TrainBiasCE={train_bias:.4f} TrainSrcCE={train_src:.4f} ValBiasCE={val_bias:.4f}")
    if val_bias < best_val_loss:
        best_val_loss = val_bias; patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

epochs = list(range(1, len(train_bias_losses)+1))
plt.figure()
plt.plot(epochs, train_bias_losses, label='Train Bias CE')
plt.plot(epochs, val_bias_losses,   label='Val Bias CE')
plt.plot(epochs, train_src_losses,  label='Train Source CE')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.tight_layout()
loss_fig = os.path.join(OUTPUT_DIR, 'loss_curves.png')
plt.savefig(loss_fig)
plt.close()
print(f"Saved loss curves → {loss_fig}")

model.eval()
true_b, pred_b, true_s, pred_s = [], [], [], []
with torch.no_grad():
    for Xb, yb, ys in loaders['test']:
        Xb = Xb.to(device)
        ob, os_ = model(Xb, grl_lambda=0.0)
        pred_b.extend(ob.argmax(1).cpu().tolist()); true_b.extend(yb.tolist())
        pred_s.extend(os_.argmax(1).cpu().tolist()); true_s.extend(ys.tolist())

bias_report = classification_report(true_b, pred_b, target_names=list(ID2BIAS.values()), digits=4)
with open(os.path.join(OUTPUT_DIR, 'bias_classification_report.txt'), 'w') as f:
    f.write(bias_report)
print(f"Saved bias report as text → {os.path.join(OUTPUT_DIR, 'bias_classification_report.txt')}")
cm_b = confusion_matrix(true_b, pred_b)
fig, ax = plt.subplots(figsize=(6,6))
disp_b = ConfusionMatrixDisplay(cm_b, display_labels=list(ID2BIAS.values()))
disp_b.plot(ax=ax, cmap='Blues')
plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
plt.setp(ax.get_yticklabels(), rotation=0)
plt.title('Bias Confusion Matrix')
plt.tight_layout()
bias_cm_fig = os.path.join(OUTPUT_DIR, 'bias_confusion_matrix.png')
plt.savefig(bias_cm_fig)
plt.close()
print(f"Saved bias confusion matrix → {bias_cm_fig}")

source_report = classification_report(true_s, pred_s, target_names=list(ID2SOURCE.values()), digits=4)
with open(os.path.join(OUTPUT_DIR, 'source_classification_report.txt'), 'w') as f:
    f.write(source_report)
print(f"Saved source report as text → {os.path.join(OUTPUT_DIR, 'source_classification_report.txt')}")
cm_s = confusion_matrix(true_s, pred_s)
fig, ax = plt.subplots(figsize=(6,6))
disp_s = ConfusionMatrixDisplay(cm_s, display_labels=list(ID2SOURCE.values()))
disp_s.plot(ax=ax, cmap='Blues')
plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
plt.setp(ax.get_yticklabels(), rotation=0)
plt.title('Source Confusion Matrix')
plt.tight_layout()
source_cm_fig = os.path.join(OUTPUT_DIR, 'source_confusion_matrix.png')
plt.savefig(source_cm_fig)
plt.close()
print(f"Saved source confusion matrix → {source_cm_fig}")
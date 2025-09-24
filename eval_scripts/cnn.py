import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

torch.set_num_threads(1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dropout=0.0):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class CNN1D(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = ConvBlock(32, 64, kernel_size=5, dropout=0.1)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = ConvBlock(64, 128, kernel_size=5, dropout=0.15)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = ConvBlock(128, 256, kernel_size=3, dropout=0.2)
        self.pool3 = nn.MaxPool1d(2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        return self.head(x)


def _compute_class_weights(y):
    values, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(values) * counts)
    class_weights = np.ones(int(values.max()) + 1, dtype=np.float32)
    for v, w in zip(values, weights):
        class_weights[int(v)] = w
    return torch.tensor(class_weights, dtype=torch.float32)


def run_cnn(
    X_train,
    y_train,
    X_test,
    task_type='regression',
    metric='accuracy',
    epochs=60,
    batch_size=128,
    lr=3e-3,
    val_size=0.25,
    num_classes=None,
    weight_decay=1e-4,
    patience=10,
):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=42,
        stratify=y_train if task_type == 'classification' else None,
    )

    X_tr = (X_tr - X_tr.mean(axis=1, keepdims=True)) / (X_tr.std(axis=1, keepdims=True) + 1e-8)
    X_val = (X_val - X_val.mean(axis=1, keepdims=True)) / (X_val.std(axis=1, keepdims=True) + 1e-8)
    X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / (X_test.std(axis=1, keepdims=True) + 1e-8)

    X_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)

    t0 = time.time()

    if task_type == 'classification':
        out_dim = num_classes if num_classes is not None else len(np.unique(y_train))
        y_tr = torch.tensor(y_tr, dtype=torch.long).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)
        if out_dim == 2:
            loss_fn = nn.CrossEntropyLoss(weight=_compute_class_weights(y_tr.cpu().numpy()).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        out_dim = 1
        y_tr = torch.tensor(y_tr, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        loss_fn = nn.MSELoss()

    model = CNN1D(out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_state = None
    best_val = float('inf')
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            xb = X_tr[i:i + batch_size]
            yb = y_tr[i:i + batch_size]
            optimizer.zero_grad()
            out = model(xb)
            if task_type == 'classification':
                loss = loss_fn(out, yb)
            else:
                loss = loss_fn(out.squeeze(1), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            if task_type == 'classification':
                val_loss = loss_fn(val_out, y_val).item()
            else:
                val_loss = loss_fn(val_out.squeeze(1), y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        tr_out = model(X_tr)
        val_out = model(X_val)
        test_out = model(X_test)

    if task_type == 'classification':
        if metric == 'auc' and (out_dim == 2):
            tr_pred = torch.softmax(tr_out, dim=1)[:, 1].detach().cpu().numpy()
            val_pred = torch.softmax(val_out, dim=1)[:, 1].detach().cpu().numpy()
            test_pred = torch.softmax(test_out, dim=1)[:, 1].detach().cpu().numpy()
            tr_score = roc_auc_score(y_tr.detach().cpu().numpy(), tr_pred)
            val_score = roc_auc_score(y_val.detach().cpu().numpy(), val_pred)
        else:
            tr_pred = tr_out.argmax(1).detach().cpu().numpy()
            val_pred = val_out.argmax(1).detach().cpu().numpy()
            test_pred = test_out.argmax(1).detach().cpu().numpy()
            tr_score = accuracy_score(y_tr.detach().cpu().numpy(), tr_pred)
            val_score = accuracy_score(y_val.detach().cpu().numpy(), val_pred)
    else:
        tr_pred = tr_out.squeeze(1).detach().cpu().numpy()
        val_pred = val_out.squeeze(1).detach().cpu().numpy()
        test_pred = test_out.squeeze(1).detach().cpu().numpy()
        tr_score = np.sqrt(mean_squared_error(y_tr.detach().cpu().numpy(), tr_pred))
        val_score = np.sqrt(mean_squared_error(y_val.detach().cpu().numpy(), val_pred))

    return {
        'train_score': float(tr_score),
        'val_score': float(val_score),
        'test_predictions': test_pred,
        'processing_time': time.time() - t0,
    }


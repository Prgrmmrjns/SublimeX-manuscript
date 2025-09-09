import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

torch.set_num_threads(1)


class CNN1D(nn.Module):
    def __init__(self, out_dim, input_size=100):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, out_dim)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def run_cnn(X_train, y_train, X_test, task_type='regression', metric='accuracy', epochs=3, batch_size=128, lr=1e-2, val_size=0.3, num_classes=None):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42, stratify=y_train if task_type=='classification' else None)
    
    # Normalize data per sample
    X_tr = (X_tr - X_tr.mean(axis=1, keepdims=True)) / (X_tr.std(axis=1, keepdims=True) + 1e-8)
    X_val = (X_val - X_val.mean(axis=1, keepdims=True)) / (X_val.std(axis=1, keepdims=True) + 1e-8)
    X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / (X_test.std(axis=1, keepdims=True) + 1e-8)
    
    X_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    
    t0 = time.time()
    
    if task_type == 'classification':
        out_dim = num_classes if num_classes is not None else len(np.unique(y_train))
        y_tr = torch.tensor(y_tr, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        loss_fn = nn.CrossEntropyLoss()
    else:
        out_dim = 1
        y_tr = torch.tensor(y_tr, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        loss_fn = nn.MSELoss()
    
    model = CNN1D(out_dim, X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, len(X_tr), batch_size):
            xb = X_tr[i:i+batch_size]
            yb = y_tr[i:i+batch_size]
            opt.zero_grad()
            out = model(xb)
            if task_type == 'classification':
                loss = loss_fn(out, yb)
            else:
                loss = loss_fn(out.squeeze(1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            if task_type == 'classification':
                val_loss = loss_fn(val_out, y_val).item()
            else:
                val_loss = loss_fn(val_out.squeeze(1), y_val).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    model.eval()
    with torch.no_grad():
        tr_out = model(X_tr)
        val_out = model(X_val)
        test_out = model(X_test)
    
    if task_type == 'classification':
        if metric == 'auc':
            tr_pred = torch.softmax(tr_out, dim=1)[:, 1].cpu().numpy()
            val_pred = torch.softmax(val_out, dim=1)[:, 1].cpu().numpy()
            test_pred = torch.softmax(test_out, dim=1)[:, 1].cpu().numpy()
            tr_score = roc_auc_score(y_tr.cpu().numpy(), tr_pred)
            val_score = roc_auc_score(y_val.cpu().numpy(), val_pred)
        else:
            tr_pred = tr_out.argmax(1).cpu().numpy()
            val_pred = val_out.argmax(1).cpu().numpy()
            test_pred = test_out.argmax(1).cpu().numpy()
            tr_score = accuracy_score(y_tr.cpu().numpy(), tr_pred)
            val_score = accuracy_score(y_val.cpu().numpy(), val_pred)
    else:
        tr_pred = tr_out.squeeze(1).cpu().numpy()
        val_pred = val_out.squeeze(1).cpu().numpy()
        test_pred = test_out.squeeze(1).cpu().numpy()
        tr_score = np.sqrt(mean_squared_error(y_tr.cpu().numpy(), tr_pred))
        val_score = np.sqrt(mean_squared_error(y_val.cpu().numpy(), val_pred))
    
    return {
        'train_score': float(tr_score),
        'val_score': float(val_score),
        'test_predictions': test_pred,
        'processing_time': time.time() - t0
    }


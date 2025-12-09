import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from params import VAL_SIZE

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
    def __init__(self, out_dim, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.3)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = self.dropout1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(torch.relu(self.bn3(self.conv3(x)))))
        return self.head(x)


def _compute_class_weights(y):
    values, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(values) * counts)
    # Create weights only for the classes that exist in the data
    class_weights = np.ones(len(values), dtype=np.float32)
    for i, (v, w) in enumerate(zip(values, weights)):
        class_weights[i] = w
    return torch.tensor(class_weights, dtype=torch.float32)


def run_cnn(
    input_series_train,
    y_train,
    input_series_test,
    task_type='regression',
    metric='accuracy',
    epochs=60,
    batch_size=128,
    lr=3e-3,
    num_classes=None,
    weight_decay=1e-4,
    patience=10,
):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

    train_features, val_features, y_train_split, y_valid = train_test_split(
        input_series_train,
        y_train,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=y_train if task_type == 'classification' else None,
    )

    # Simple normalization instead of per-sample normalization
    train_mean = train_features.mean()
    train_std = train_features.std()
    train_features = (train_features - train_mean) / (train_std + 1e-8)
    val_features = (val_features - train_mean) / (train_std + 1e-8)
    test_features = (input_series_test - train_mean) / (train_std + 1e-8)

    # Auto-detect number of channels from input shape
    # Assumes input is concatenated time series: if we have N series of length T, input has N*T columns
    # For REMC: 5 series * 100 timepoints = 500 columns → 5 channels
    # For AZT1D: 3 series * 24 timepoints = 72 columns → 3 channels
    # For Bonn EEG: 1 series * 4101 timepoints = 4101 columns → 1 channel
    total_features = train_features.shape[1]
    
    # Determine number of channels (heuristic: find common divisors)
    if total_features % 100 == 0:  # REMC case: 5 series of 100 timepoints
        n_channels = total_features // 100
    elif total_features % 24 == 0:  # AZT1D case: 3 series of 24 timepoints
        n_channels = total_features // 24
    elif total_features > 1000:  # Large univariate time series (like Bonn EEG)
        n_channels = 1
    else:  # Fallback: treat as single channel
        n_channels = 1
    
    # Reshape for multi-channel input: (batch, n_channels, time_points)
    train_features = torch.tensor(train_features, dtype=torch.float32).reshape(train_features.shape[0], n_channels, -1).to(device)
    val_features = torch.tensor(val_features, dtype=torch.float32).reshape(val_features.shape[0], n_channels, -1).to(device)
    test_features = torch.tensor(test_features, dtype=torch.float32).reshape(test_features.shape[0], n_channels, -1).to(device)

    if task_type == 'classification':
        out_dim = num_classes if num_classes is not None else len(np.unique(y_train))
        y_train_split = torch.tensor(y_train_split, dtype=torch.long).to(device)
        y_valid = torch.tensor(y_valid, dtype=torch.long).to(device)
        if out_dim == 2:
            loss_fn = nn.CrossEntropyLoss(weight=_compute_class_weights(y_train_split.cpu().numpy()).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        out_dim = 1
        y_train_split = torch.tensor(y_train_split, dtype=torch.float32).to(device)
        y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
        loss_fn = nn.MSELoss()

    model = CNN1D(out_dim, in_channels=n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_state = None
    best_val = float('inf')
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(0, len(train_features), batch_size):
            xb = train_features[i:i + batch_size]
            yb = y_train_split[i:i + batch_size]
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
            val_out = model(val_features)
            if task_type == 'classification':
                val_loss = loss_fn(val_out, y_valid).item()
            else:
                val_loss = loss_fn(val_out.squeeze(1), y_valid).item()

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
        test_out = model(test_features)

    if task_type == 'classification':
        if metric == 'auc' and (out_dim == 2):
            test_pred = torch.softmax(test_out, dim=1)[:, 1].detach().cpu().numpy()
        else:
            test_pred = test_out.argmax(1).detach().cpu().numpy()
    else:
        test_pred = test_out.squeeze(1).detach().cpu().numpy()
    
    # Clean up to prevent memory leaks
    del model, optimizer, scheduler, train_features, val_features, test_features
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return test_pred
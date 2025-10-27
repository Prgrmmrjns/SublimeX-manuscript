import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class LightGBMModelWrapper:
    def __init__(self, task_type='classification', n_classes=None):
        self.task_type = task_type
        self.n_classes = n_classes
        lgb_params = {
            'num_threads': 1,
            'verbosity': -1,
            'max_depth': 3,
            'n_estimators': 1000,
            'learning_rate': 0.1,
        }
        self.model = lgb.LGBMClassifier(**lgb_params) if task_type == 'classification' else lgb.LGBMRegressor(**lgb_params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        else:
            self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.task_type == 'classification':
            preds = self.model.predict_proba(X)
            return preds[:, 1] if self.n_classes == 2 else preds
        return self.model.predict(X)
    
    def clone(self):
        return LightGBMModelWrapper(self.task_type, self.n_classes)

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

class NeuralNetworkWrapper:
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.3, epochs=100, lr=0.001, batch_size=32):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        # Force CPU to avoid MPS segmentation faults
        self.device = torch.device('cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training neural network with input size: {self.input_size}")
        print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
        
        self.model = SimpleNeuralNetwork(self.input_size, self.hidden_sizes, self.dropout).to(self.device)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        else:
            X_val_tensor = None
            y_val_tensor = None
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for i in range(0, len(X_train_tensor), self.batch_size):
                batch_X = X_train_tensor[i:i + self.batch_size]
                batch_y = y_train_tensor[i:i + self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Clean up tensors to free memory
        del X_train_tensor, y_train_tensor
        if X_val_tensor is not None:
            del X_val_tensor, y_val_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze().cpu().numpy()
        
        return preds.reshape(-1, 1) if preds.ndim == 0 else preds
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def clone(self):
        return NeuralNetworkWrapper(self.input_size, self.hidden_sizes, self.dropout, self.epochs, self.lr, self.batch_size)

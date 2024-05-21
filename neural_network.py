import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torcheval.metrics import R2Score
import numpy as np
import itertools
import time
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Hardcoded hyperparameters
optimiser_names = ['Adagrad', 'AdamW', 'Adam', 'SGD']
learning_rates = [0.001, 0.004, 0.007, 0.01]
hidden_layer_widths = [16]  # Single value as a list
depths = [1]  # Single value as a list

start_time = time.time()
data = pd.read_csv('tabular_data/clean_tabular_data.csv')
data = pd.DataFrame(data)
data = data.drop(columns=['ID', 'Category', 'Title', 'Description', 'Amenities', 'Location', 'url'])

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        item = self.data.iloc[index]
        features = torch.tensor(item[:-1], dtype=torch.float)
        label = torch.tensor(item[3], dtype=torch.float)
        return features, label

    def __len__(self):
        return len(self.data)

dataset = AirbnbNightlyPriceRegressionDataset()

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=13, shuffle=True)
test_loader = DataLoader(test_data, batch_size=13, shuffle=True)
val_loader = DataLoader(val_data, batch_size=13, shuffle=False)

class NN(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_width, depth):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_layer_width))
        for variable in range(depth - 1):
            self.layers.append(torch.nn.Linear(hidden_layer_width, hidden_layer_width))
        self.layers.append(torch.nn.Linear(hidden_layer_width, 1))

    def forward(self, X):
        for layer in self.layers[:-1]:
            X = F.relu(layer(X))
        X = self.layers[-1](X)
        return X

input_size = len(data.columns) - 1

def train(model, train_loader, val_loader, optimiser_name, learning_rate, epochs=10):
    if optimiser_name == 'Adagrad':
        optimiser = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimiser_name == 'AdamW':
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimiser_name == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimiser_name == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()
    batch_idx = 0
    r2_metric = R2Score()

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            optimiser.zero_grad()
            prediction = model(features)
            loss = F.mse_loss(prediction, labels.unsqueeze(1))
            loss.backward()
            optimiser.step()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
            print(loss.item())
            train_rsme_loss = np.sqrt(loss.item())
            r2_metric.update(prediction, labels.unsqueeze(1))
            train_r2_loss = r2_metric.compute()

        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_features, val_labels = val_batch
                val_prediction = model(val_features)
                val_loss += F.mse_loss(val_prediction, val_labels.unsqueeze(1))
        val_loss /= len(val_loader)
        val_rmse = np.sqrt(val_loss.item())
        r2_metric.update(val_prediction, val_labels.unsqueeze(1))
        val_r2_loss = r2_metric.compute()

        test_loss = 0.0
        with torch.no_grad():
            for test_batch in test_loader:
                test_features, test_labels = test_batch
                test_prediction = model(test_features)
                test_loss += F.mse_loss(test_prediction, test_labels.unsqueeze(1))
        test_loss /= len(test_loader)
        test_rmse = np.sqrt(test_loss.item())
        r2_metric.update(test_prediction, test_labels.unsqueeze(1))
        test_r2_loss = r2_metric.compute()

        print(f"Epoch {epoch + 1}/{epochs}, Training RSME Loss: {train_rsme_loss}, Validation RMSE Loss: {val_rmse}, Test RMSE Loss: {test_rmse}, Train r2 Loss:{train_r2_loss}, Validation r2 Loss:{val_r2_loss}, Test r2 Loss:{test_r2_loss}")

    metrics = {'RMSE_loss': {'train': train_rsme_loss, 'validation': val_rmse, 'test': test_rmse},
               'R_squared': {'train': train_r2_loss.item(), 'validation': val_r2_loss.item(), 'test': test_r2_loss.item()},
               'training_duration': time.time() - start_time}

    return metrics

def save_model(model, hyperparameters, metrics, model_type='torch'):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'models/neural_networks/regression/{current_time}'
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

def find_best_nn():
    best_rmse = float('inf')
    best_hyperparameters = {}
    best_metrics = {}

    for optimiser_name, learning_rate, hidden_layer_width, depth in itertools.product(optimiser_names, learning_rates, hidden_layer_widths, depths):
        hyperparameters = {
            'optimiser_name': optimiser_name,
            'learning_rate': learning_rate,
            'hidden_layer_width': hidden_layer_width,
            'depth': depth
        }

        model = NN(input_size, hidden_layer_width, depth)
        metrics = train(model, train_loader, val_loader, optimiser_name, learning_rate)

        if metrics['RMSE_loss']['validation'] < best_rmse:
            best_rmse = metrics['RMSE_loss']['validation']
            best_hyperparameters = hyperparameters
            best_metrics = metrics

    print("Best hyperparameters:", best_hyperparameters)
    print("Best RMSE:", best_rmse)

    save_model(model, best_hyperparameters, best_metrics)

find_best_nn()
end_time = time.time()
training_duration = end_time - start_time
print(f"Training duration: {training_duration} seconds")

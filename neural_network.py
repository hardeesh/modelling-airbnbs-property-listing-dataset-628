import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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

train_loader = DataLoader(dataset, batch_size=13, shuffle=True)
test_loader = DataLoader(dataset, batch_size=13, shuffle=True)
val_loader = DataLoader(val_data, batch_size=13, shuffle=False)

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1) 
        
    def forward(self, features):
        return self.linear(features)

input_size = len(data.columns) - 1 
model = LinearRegression(input_size)

def train(model, train_loader, val_loader, epochs=10):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.2)

    writer = SummaryWriter()

    batch_idx = 0

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

        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_features, val_labels = val_batch
                val_prediction = model(val_features)
                val_loss += F.mse_loss(val_prediction, val_labels.unsqueeze(1)).item()
        val_loss /= len(val_loader)
        val_rmse = np.sqrt(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}, Validation RMSE: {val_rmse}")
                    

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(11, 16)
        self.linear_layer2 = torch.nn.Linear(16, 1)

    def forward(self, X):
        X = self.linear_layer(X)
        X = F.relu(X)
        X = self.linear_layer2(X)
        return X

        
model = NN()

train(model, train_loader, val_loader)
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

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

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset, batch_size=4, shuffle=True)

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1) 
        
    def forward(self, features):
        return self.linear(features)

input_size = len(data.columns) - 1 
model = LinearRegression(input_size)

def train(model, train_loader, epochs=10):
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels.unsqueeze(1))
            loss.backward()
            print(loss)
            #optimisation step
            break

train(model, train_loader)
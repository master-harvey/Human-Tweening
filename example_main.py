#Copied from the pytorch examples: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html?highlight=lstm
import torch
import torch.nn as nn
import torch.optim as optim
import json

class MouseMovementModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MouseMovementModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])  
        return out

def load_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def train_model(model, data, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for sample in data:
            input_vector = torch.tensor([sample['translation']['x'], sample['translation']['y'], sample['translation']['t']], dtype=torch.float32)
            input_vector = input_vector.unsqueeze(0).unsqueeze(0)  # Add two extra dimensions
            target_path = torch.tensor([[step['x'], step['y'], step['t']] for step in sample['steps']], dtype=torch.float32)
            
            optimizer.zero_grad()
            output = model(input_vector)
            loss = criterion(output, target_path)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    data = load_data('preprocessed_training_data.json')
    model = MouseMovementModel(input_size=3, hidden_size=50, output_size=3, num_layers=2)
    train_model(model, data)
import torch
import torch.nn as nn

class HumanTweeningNN(nn.Module):
    def __init__(self):
        super(HumanTweeningNN, self).__init__()
        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output size should match the translation vector size
        
        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Pass through LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the last time step output
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)  # Final output
        return x

# Example of how to instantiate and use the model
if __name__ == "__main__":
    model = HumanTweeningNN()
    print(model)

    # Example input: batch size of 1, sequence length of 10, input size of 2 (x, y coordinates)
    example_input = torch.randn(1, 10, 2)
    output = model(example_input)
    print("Output shape:", output.shape)  # Should be [1, 2]
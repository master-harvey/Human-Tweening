import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, output_size=3, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pack sequences and apply LSTM
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_input, (h0, c0))
        
        # Unpack the packed sequence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # Apply fully connected layer
        out = self.fc(out)
        
        return out

    def train_model(self, train_data, num_epochs=100, learning_rate=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for sample in train_data:
                translation = sample['translation']
                steps = sample['steps']
                
                # Scale inputs to [-1, 1]
                scaled_x = torch.tensor(translation['x'] / 1920.0, dtype=torch.float32)
                scaled_y = torch.tensor(translation['y'] / 1080.0, dtype=torch.float32)
                scaled_t = torch.tensor(translation['t'], dtype=torch.float32)  # Assuming 't' is already scaled correctly
                
                # Stack scaled inputs into a single tensor
                inputs = torch.stack([scaled_x, scaled_y, scaled_t], dim=0).unsqueeze(0).to(device)
                
                # Prepare target tensor
                target = torch.tensor([[step['x'] / 1920.0, step['y'] / 1080.0, step['t']] for step in steps], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Prepare input lengths for pack_padded_sequence
                lengths = torch.tensor([target.size(1)]).to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs, lengths)
                
                # Ensure the output length matches the target length
                outputs = outputs[:, :target.size(1), :]
                
                loss = criterion(outputs.squeeze(0), target.squeeze(0))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_data)
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        torch.save(self.state_dict(), 'model.pth')
        print('Model saved to model.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        
    def inference(self, input_vector):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        # Scale input vector
        scaled_x = torch.tensor(input_vector[0] / 1920.0, dtype=torch.float32).unsqueeze(0)
        scaled_y = torch.tensor(input_vector[1] / 1080.0, dtype=torch.float32).unsqueeze(0)
        scaled_t = torch.tensor(input_vector[2], dtype=torch.float32).unsqueeze(0)  # Assuming 't' is already scaled correctly
        
        input_vector_scaled = torch.cat([scaled_x, scaled_y, scaled_t], dim=1).unsqueeze(0).to(device)
        
        output_sequence = []
        accumulated_sum = torch.zeros(3).to(device)
        while True:
            lengths = torch.tensor([1]).to(device)
            output = self(input_vector_scaled, lengths)
            step = output.squeeze(0).squeeze(0)
            output_sequence.append(step.tolist())
            accumulated_sum += step
            if torch.allclose(accumulated_sum, input_vector_scaled.squeeze(0), atol=1e-5):
                break
            input_vector_scaled = step.unsqueeze(0).unsqueeze(0)
        
        # Reverse scale output_sequence to pixel values
        output_sequence_pixel = []
        for step in output_sequence:
            output_x = step[0] * 1920.0
            output_y = step[1] * 1080.0
            output_t = step[2]  # Assuming 't' does not need scaling back
            
            output_sequence_pixel.append({'x': output_x.item(), 'y': output_y.item(), 't': output_t.item()})
        
        return output_sequence_pixel


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Train the model
        if not os.path.exists("preprocessed_training_data.json"):
            print("Training data file not found!")
            sys.exit(1)
        
        with open("preprocessed_training_data.json", 'r') as f:
            training_data = json.load(f)
        
        model = LSTMModel()
        model.train_model(training_data, num_epochs=10)
    else:
        # Run inference
        input_vector = tuple(map(int, sys.argv[1].strip('()').split(',')))
        
        model = LSTMModel()
        model.load_model('model.pth')
        
        output_sequence = model.inference(input_vector)
        print(f'Input: {input_vector}')
        print(f'Output Sequence: {output_sequence}')

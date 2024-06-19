import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

class VectorDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.src_data = [item['src'] for item in data]
        self.trg_data = [item['trg'] for item in data]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = self.src_data[idx]
        trg = self.trg_data[idx]
        return torch.tensor(src, dtype=torch.float32), torch.tensor(trg, dtype=torch.float32)

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(self.dropout(src))
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.rnn(self.dropout(input.unsqueeze(1)), (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = trg.shape[2]

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        hidden, cell = self.encoder(src)
        input = trg[:, 0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t, :] if teacher_force else output

        return outputs

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, trg = batch
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1, output_dim)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

if __name__ == "__main__":
    INPUT_DIM = 3  # Dimension of the input vector (e.g., 100, 200)
    OUTPUT_DIM = 3  # Dimension of the output vectors (e.g., 1, 2)
    HID_DIM = 128
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()  # Adjust as necessary for your noise tolerance and summing requirements

    N_EPOCHS = 10
    CLIP = 1

    train_iterator = DataLoader(VectorDataset("preprocessed_training_data.json"), batch_size=32, shuffle=True)


    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')

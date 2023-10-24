import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import *
import torchaudio

class MusicDataset(Dataset):
    def __init__(self, files, sample_length):
        self.files = files
        self.sample_length = sample_length

    def __len__(self):
        return len(self.files)

    def load_and_normalize(filename):
        waveform, sample_rate = torchaudio.load(filename)
        
        # Normalize to have values between -1 and 1
        waveform = torch.mean(waveform, dim=0)
        waveform = waveform / torch.max(torch.abs(waveform))

        return waveform

    def __getitem__(self, idx):
        waveform = self.load_and_normalize(self.files[idx])

        if waveform.shape[0] < self.sample_length: # zero-padding
            waveform = torch.nn.functional.pad(waveform, (0, self.sample_length - waveform.shape[0]))

        return waveform[:self.sample_length]

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        out = self.transformer(x)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    loss = nn.MSELoss()
    model = TransformerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
    for idx, sample in enumerate(loader):
        # Forward pass
        output = model(sample)
        loss = loss_fn(output, sample)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1000 == 0:
            print('Loss: ', loss.item())
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import os

def collate_fn(batch, max_length = 4800000):
    mono_batch = [x.mean(dim=0, keepdim=True) for x in batch]
    padded_batch = [torch.nn.functional.pad(x, (0, max_length - x.size(-1))) for x in mono_batch]
    
    return torch.stack(padded_batch, dim=0)

class MusicDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the music subdirectories.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []

        for artist in os.listdir(root_dir):
            artist_path = os.path.join(root_dir, artist)
            if os.path.isdir(artist_path):
                for file in os.listdir(artist_path):
                    if file.lower().endswith(('.wav')):
                        self.file_paths.append(os.path.join(artist_path, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform
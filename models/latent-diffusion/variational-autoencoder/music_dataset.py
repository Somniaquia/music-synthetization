from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import os

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
                    if file.lower().endswith(('.mp4', '.aac')):
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

        sample = {'waveform': waveform, 'sample_rate': sample_rate, 'audio_path': audio_path}
        return sample
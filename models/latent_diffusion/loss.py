import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import torchaudio

class TimeDomainLoss(nn.Module):
    def __init__(self):
        super(TimeDomainLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.l1_loss(y_pred, y_true)

class FrequencyDomainLoss(nn.Module):
    def __init__(self):
        super(FrequencyDomainLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        # Compute the STFT of y_pred and y_true
        stft_pred = torch.stft(y_pred.squeeze(), n_fft=1024, hop_length=256, win_length=1024, return_complex=True)
        stft_true = torch.stft(y_true.squeeze(), n_fft=1024, hop_length=256, win_length=1024, return_complex=True)

        # Compute the magnitude of the STFT
        mag_pred = stft_pred.abs()
        mag_true = stft_true.abs()

        # Compute the L1 loss between the magnitudes
        loss = self.l1_loss(mag_pred, mag_true)
        return loss
    
class CombinedAudioLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedAudioLoss, self).__init__()
        self.time_domain_loss = TimeDomainLoss()
        self.frequency_domain_loss = FrequencyDomainLoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        time_loss = self.time_domain_loss(y_pred, y_true)
        freq_loss = self.frequency_domain_loss(y_pred, y_true)
        loss = (1 - self.alpha) * time_loss + self.alpha * freq_loss
        return loss
    
if __name__ == "__main__":
    def get_random_file_in_subfolders(directory):
        all_files = []
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                all_files.append(os.path.join(foldername, filename))

        if all_files:
            return random.choice(all_files)
        else:
            return None

    path = get_random_file_in_subfolders("data/8kHz_8bit")
    path_2 = get_random_file_in_subfolders("data/8kHz_8bit")

    waveform_1, sample_rate_1 = torchaudio.load(path)
    waveform_2, sample_rate_2 = torchaudio.load(path_2)

    min_length = min(waveform_1.size(1), waveform_2.size(1))
    waveform_1 = waveform_1[:, :min_length]
    waveform_2 = waveform_2[:, :min_length]

    combined_loss = CombinedAudioLoss(alpha=0.5)

    loss = combined_loss(waveform_1, waveform_2)
    print('Combined Audio Loss:', loss.item())
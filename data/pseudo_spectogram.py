import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pygame.mixer
import random
import os

from scipy.ndimage import gaussian_filter

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

pygame.mixer.init()
y, sr = librosa.load(path)
y_2, sr_2 = librosa.load(path_2, sr=sr)
pygame.mixer.music.load(path)
pygame.mixer.music.play(loops=-1)

min_length = min(len(y), len(y_2))
y = y[:min_length]
y_2 = y_2[:min_length]

mixed_y = (y + y_2) / 2.0

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mixed_S = librosa.feature.melspectrogram(y=mixed_y, sr=sr, n_mels=128)

log_S = librosa.amplitude_to_db(S, ref=np.max)
log_mixed_S = librosa.amplitude_to_db(mixed_S, ref=np.max)

# Apply Gaussian blur to the spectrogram
sigma = 2  # Standard deviation of the Gaussian kernel
blurred_log_S = gaussian_filter(log_S, sigma=sigma, radius=10)

duration_in_frames = log_S.shape[1]
segment_length = duration_in_frames // 3
times = librosa.frames_to_time(np.arange(duration_in_frames), sr=sr)

plt.figure(figsize=(10, 8))
plt.rcParams["font.family"] = 'Microsoft YaHei'
# plt.suptitle(path.split('\\')[-1].split('.wav')[0], fontsize=16)

for i in range(3):
    plt.subplot(3, 2, 2*i+1)
    start_frame = i * segment_length
    end_frame = (i+1) * segment_length if i != 2 else duration_in_frames
    librosa.display.specshow(log_S[:, start_frame:end_frame], sr=sr, x_axis='time', y_axis='mel', x_coords=times[start_frame:end_frame], cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Segment {i+1} - Original')

    sigma_x = random.uniform(0.5, 2)  # Random standard deviation for horizontal blurring
    sigma_y = 2  # Standard deviation for vertical blurring
    blurred_log_S = gaussian_filter(log_S[:, start_frame:end_frame], sigma=[sigma_y, sigma_x])

    plt.subplot(3, 2, 2*i+2)
    # Apply random horizontal blur
    sigma_x = random.uniform(0.5, 2)  # Random standard deviation for horizontal blurring
    sigma_y = 2  # Standard deviation for vertical blurring
    blurred_log_mixed_S = gaussian_filter(log_mixed_S[:, start_frame:end_frame], sigma=[sigma_y, sigma_x])

    librosa.display.specshow(blurred_log_mixed_S, sr=sr, x_axis='time', y_axis='mel', x_coords=times[start_frame:end_frame], cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Segment {i+1} - Reconstructed')
    
plt.tight_layout()
plt.show()

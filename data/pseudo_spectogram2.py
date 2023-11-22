import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import random

def get_random_file_in_subfolders(directory):
    all_files = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                all_files.append(os.path.join(foldername, filename))
                
    if all_files:
        return random.choice(all_files)
    else:
        return None

def load_and_mix_tracks(directory, sr=22050):
    paths = [get_random_file_in_subfolders(directory) for _ in range(2)]
    ys = [librosa.load(path, sr=sr)[0] for path in paths]
    
    min_length = min(len(y) for y in ys)
    ys = [y[:min_length] for y in ys]
    
    x = np.linspace(0, np.pi, min_length)
    cosine_weights = (np.cos(x/100) + 1) / 2
    
    mixed_y = ys[0] * cosine_weights + ys[1] * (1 - cosine_weights)
    #mixed_y = mixed_y * cosine_weights + ys[2] * (1 - cosine_weights)
    
    return mixed_y, sr, paths

directory = "data/8kHz_8bit"
y, sr, paths = load_and_mix_tracks(directory)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_S = librosa.amplitude_to_db(S, ref=np.max)

# Apply extreme Gaussian blur to the spectrogram
sigma_x = random.uniform(1, 2)  # Random standard deviation for horizontal blurring
sigma_y = 3  # Standard deviation for vertical blurring
blurred_log_S = gaussian_filter(log_S, sigma=[sigma_y, sigma_x]) * 0.5 + np.random.randn(*log_S.shape) * 0.5

duration_in_frames = log_S.shape[1]
segment_length = duration_in_frames // 3
times = librosa.frames_to_time(np.arange(duration_in_frames), sr=sr)

plt.figure(figsize=(10, 8))
plt.rcParams["font.family"] = 'Microsoft YaHei'
plt.suptitle("Synthesized sample", fontsize=16)

for i in range(3):
    plt.subplot(3, 1, i+1)
    start_frame = i * segment_length
    end_frame = (i+1) * segment_length if i != 2 else duration_in_frames
    librosa.display.specshow(blurred_log_S[:, start_frame:end_frame], sr=sr, x_axis='time', y_axis='mel', x_coords=times[start_frame:end_frame], cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Segment {i+1}')

plt.tight_layout()
plt.show()

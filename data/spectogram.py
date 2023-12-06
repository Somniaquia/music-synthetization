import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pygame.mixer
import random
import os

def get_random_file_in_subfolders(directory):
    all_files = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            all_files.append(os.path.join(foldername, filename))

    if all_files:
        return random.choice(all_files)
    else:
        return None

def plot_spectogram(y, sr):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(loops=-1)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)

    duration_in_frames = log_S.shape[1]
    segment_length = duration_in_frames // 3
    times = librosa.frames_to_time(np.arange(duration_in_frames), sr=sr)

    plt.figure(figsize=(10, 8))
    plt.rcParams["font.family"] = 'Microsoft YaHei'
    plt.suptitle(path.split('\\')[-1].split('.wav')[0], fontsize=16)

    for i in range(3):
        plt.subplot(3, 1, i+1)
        start_frame = i * segment_length
        end_frame = (i+1) * segment_length if i != 2 else duration_in_frames
        librosa.display.specshow(log_S[:, start_frame:end_frame], sr=sr, x_axis='time', y_axis='mel', x_coords=times[start_frame:end_frame], cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Segment {i+1}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = "C:\Somnia\Projects\music-synthetization\data\8kHz_8bit\ぬゆり\フラジール - GUMI _ Fragile - nulut.wav"
    y, sr = librosa.load(path)
    plot_spectogram(y, sr)
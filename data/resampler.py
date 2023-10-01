import os
from pydub import AudioSegment
import librosa
import soundfile as sf
from pathlib import Path

def resample_audio(src_path, dst_path, target_sample_rate=8000):
    y, sr = librosa.load(src_path, sr=None)
    y_resampled = librosa.resample(y, sr, target_sample_rate)
    sf.write(dst_path, y_resampled, target_sample_rate, subtype='PCM_24')

def convert_and_resample(root_folder, target_sample_rate=8000):
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            src_filepath = subdir + os.sep + file

            if src_filepath.endswith(".mp4"):
                dst_filepath = (os.path.splitext(src_filepath)[0] + '.wav').replace('raw', str(target_sample_rate).replace('000', 'kHz'))
                Path(os.sep.join(dst_filepath.split(os.sep)[:-1])).mkdir(parents=True, exist_ok=True)

                if os.path.isfile(dst_filepath):
                    continue

                audio = AudioSegment.from_file(src_filepath, format="mp4")
                audio.export(dst_filepath, format="wav") 

                resample_audio(dst_filepath, dst_filepath, target_sample_rate)

                os.remove(src_filepath) 

convert_and_resample('data\\raw', 16000)
import os
from pydub import AudioSegment
from pathlib import Path

def convert_and_resample(root_folder, target_sample_rate=8000, bit_depth=16):
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            src_filepath = os.path.join(subdir, file)

            if src_filepath.endswith(".mp4"):
                dst_filepath = src_filepath.replace('raw', str(target_sample_rate).replace('000', 'kHz') + '_' + str(bit_depth) + 'bit')

                Path(os.path.dirname(dst_filepath)).mkdir(parents=True, exist_ok=True)

                if os.path.isfile(dst_filepath):
                    print("File already exists: " + dst_filepath)
                    continue

                try:
                    audio = AudioSegment.from_file(src_filepath, format="mp4")
                    audio = audio.set_frame_rate(target_sample_rate)
                    audio = audio.set_sample_width(bit_depth // 8)
                    audio.export(dst_filepath.replace('mp4', 'wav'), format="wav") 
                    print("Saved to " + dst_filepath)
                except Exception as e:
                    print("Failed to process {}: {}".format(src_filepath, str(e)))

convert_and_resample('data\\raw', 8000, 8)
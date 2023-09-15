import pytube

def download_video(video_url, save_location, audio_only=True):
    print(video_url)
    yt = pytube.YouTube(video_url)
    print(yt.age_restricted)
    video_title = yt.title

    audio_stream = yt.streams.filter(only_audio=audio_only, file_extension='mp4').first()
    audio_stream.download(filename=f"data/audio/{save_location}/{video_title}.mp4")

    return video_title

def evaluate(filename):
    import audioowl

    import matplotlib.pyplot as plt
    waveform = audioowl.get_waveform(filename, sr=22050)
    data = audioowl.analyze_file(filename, sr=22050)

    plt.figure()
    plt.vlines(data['beat_samples'], -1.0, 1.0)
    plt.plot(waveform)
    plt.show()
    
if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")
    filename = download_video(video_url, "")
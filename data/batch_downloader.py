import os
import re
import pytube
import musicbrainzngs
from youtubesearchpython import VideosSearch

musicbrainzngs.set_useragent("MusicBrainzAPI", "0.1", contact="Somniaquia@gmail.com")

def download_video(video_url, save_location, audio_only=True):
    yt = pytube.YouTube(video_url)
    # video_title = re.sub(r'[^\x00-\x7F]', '_', yt.title)
    video_title = re.sub(r'[\x00-\x1F\x7F-\x9F\/:?*<>|]', '_', yt.title)
    video_title = re.sub(r'_+', '_', video_title)

    path = f"data/raw/{save_location}"
    if not os.path.exists(path):
        os.makedirs(path)

    audio_stream = yt.streams.filter(only_audio=audio_only, file_extension='mp4').first()
    audio_stream.download(filename=f"{path}/{video_title}.mp4")

    return video_title

def search_artist_songs(artist_name):
    try:
        artist_search = musicbrainzngs.search_artists(artist_name)
        if "artist-list" not in artist_search or not artist_search["artist-list"]:
            print("Artist not found.")
            return

        artist_name = artist_search["artist-list"][0]["name"]
        artist_id = artist_search["artist-list"][0]["id"]

        append_link(artist_name)

        releases = musicbrainzngs.browse_releases(artist=artist_id)

        for release in releases["release-list"]:
            release_id = release["id"]

            release_info = musicbrainzngs.get_release_by_id(release_id, includes=["recordings"])['release']

            if "medium-count" not in release_info or release_info["medium-count"] == 0:
                print(release_info)
                print("No medium found for this release.\n")
                continue

            medium_list = release_info["medium-list"][0]  # Assuming one medium per release

            if "track-list" not in medium_list:
                print("No tracks found for this release.")
                continue

            # Iterate through the tracks on the medium
            for track in medium_list["track-list"]:
                track_title = track["recording"]["title"]

                youtube_search = VideosSearch(f"{artist_name} {track_title} original")
                results = youtube_search.result()['result']

                if results:
                    youtube_link = results[0]["link"]
                    print(f"Track found: {track_title} - {artist_name}, Link: {youtube_link}")

                    try:
                        if not is_link_already_saved(youtube_link):
                            download_video(youtube_link, artist_name)
                            print("Download succeed! \n")
                            append_link(youtube_link)
                        else:
                            print("Skipping song as it is already downloaded \n")
                    except Exception as e:
                        print(e)

                else:
                    print(f"Track: {track_title} by {artist_name}")
                    print("YouTube Link not found.")

    except musicbrainzngs.WebServiceError as exc:
        print(f"MusicBrainz API error: {exc}")

def is_link_already_saved(link):
    if os.path.isfile("data/links.txt"):
        with open("data/links.txt", "r") as file:
            return link in file.read()
    return False

def append_link(link):
    with open("data/links.txt", "a") as file:
        file.write(link + "\n")

if __name__ == "__main__":
    while True:
        artist_names = input("Enter the artists' names to download albums of (seperated with commas): ")
        for name in artist_names.split(","):
            search_artist_songs(name)
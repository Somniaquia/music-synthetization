import os
import musicbrainzngs
from youtubesearchpython import VideosSearch

from downloader import download_video

# Set your MusicBrainz API credentials
musicbrainzngs.set_useragent("MusicBrainzAPI", "0.1", contact="Somniaquia@gmail.com")

def search_artist_songs(artist_name):
    try:
        artist_search = musicbrainzngs.search_artists(artist_name)
        if "artist-list" not in artist_search or not artist_search["artist-list"]:
            print("Artist not found.")
            return

        artist_name = artist_search["artist-list"][0]["name"]
        artist_id = artist_search["artist-list"][0]["id"]

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

                youtube_search = VideosSearch(f"{artist_name} {track_title} audio")
                results = youtube_search.result()['result']

                if results:
                    youtube_link = results[0]["link"]
                    print(f"Track found: {track_title} - {artist_name}, Link: {youtube_link}")

                    try:
                        download_video(youtube_link, artist_name)
                        print("Download succeed! \n")
                        append_link(youtube_link)
                    except:
                        print("The video wasn't downloaded.")

                else:
                    print(f"Track: {track_title} by {artist_name}")
                    print("YouTube Link not found.")

    except musicbrainzngs.WebServiceError as exc:
        print(f"MusicBrainz API error: {exc}")

links_file = "data/links"

def is_link_already_saved(link):
    if os.path.isfile(file):
        with open(file, "r") as file:
            return link in file.read()
    return False

def append_link(link):
    with open(file, "a") as file:
        file.write(link + "\n")

if __name__ == "__main__":
    artist_name = input("Enter the artist's name: ")
    print("\n")
    search_artist_songs(artist_name)

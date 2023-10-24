import os
import re
import requests
import json

def get_channel_name(channel_id, api_key):
    endpoint = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={channel_id}&key={api_key}"
    response = requests.get(endpoint).json()
    return response['items'][0]['snippet']['title']


def get_video_name(video_id, api_key):
    endpoint = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
    response = requests.get(endpoint).json()
    return response['items'][0]['snippet']['title']


def resolve_unusable_characters(title):
    title = re.sub(r'[\x00-\x1F\x7F-\x9F\/:?*<>|# ]', '_', title)
    title = re.sub(r'_+', '_', title)

    return title


def download_thumbnail(video_id, save_path):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    
    response = requests.get(thumbnail_url, stream=True)
    response.raise_for_status()
    
    if not os.path.exists(save_path.split('/')[0]):
        os.makedirs(save_path.split('/'[0]))

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Thumbnail saved to {save_path}")


def get_channel_videos(channel_id, api_key):
    base_url = "https://www.googleapis.com/youtube/v3"
    endpoint = f"{base_url}/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50"
    video_links = []

    while True:
        response = requests.get(endpoint)
        if response.status_code != 200:
            raise Exception("API request failed with status code " + str(response.status_code))
        
        result = response.json()

        for item in result.get("items", []):
            if item["id"]["kind"] == "youtube#video":
                video_id = item["id"]["videoId"]
                download_thumbnail(video_id, f"{get_channel_name(channel_id, api_key)}/{resolve_unusable_characters(get_video_name(video_id, api_key))}.jpg")
                video_links.append(f"https://www.youtube.com/watch?v={video_id}")

        if "nextPageToken" in result:
            endpoint += "&pageToken=" + result["nextPageToken"]
        else:
            break

    return video_links

if __name__ == "__main__":
    API_KEY = "AIzaSyDHxBwWRiSfkKijUkwNWARgOquqwb4LZpY"
    CHANNEL_ID = "UCmMxEFwIOMGGoThkmtZZOvQ"
    
    get_channel_videos(CHANNEL_ID, API_KEY)
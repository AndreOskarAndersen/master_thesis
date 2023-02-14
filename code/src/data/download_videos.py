import pandas as pd
from pytube import YouTube
from tqdm import tqdm
from global_variables import *

def _download_video(video_id):
    url = "https://www.youtube.com/watch?v=" + video_id
    save_path = OUTPUT_PATH + CORPUS_FOLDERS["videos_folder"] + video_id
    YouTube(url).streams.filter(file_extension="mp4", res="1080p").first().download(save_path)

def download_videos():
    print("Downloading videos...")
    csv = pd.read_csv(OUTPUT_PATH + METAINFO_NAME)
    video_ids = csv["video_id"].to_list()
    
    for video_id in tqdm(video_ids, desc="Downloading videos", leave=False):
        _download_video(video_id)
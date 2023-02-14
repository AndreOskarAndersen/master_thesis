import pandas as pd
from pytube import YouTube
from tqdm import tqdm
from global_variables import *

def _download_video(video_id: str):
    """
    Function for downloading a youtube-video, given its video-ID.
    
    Parameters
    ----------
    video_id : str
        The ID of the video to download
    """
    
    # Url of the video to download
    url = "https://www.youtube.com/watch?v=" + video_id
    
    # Path of where to save the downloaded video
    save_path = OUTPUT_PATH + CORPUS_FOLDERS["videos_folder"] + video_id
    
    # Downloading the video using the .mp4-format in the 1920x1080-resolution.
    YouTube(url).streams.filter(file_extension="mp4", res="1080p").first().download(save_path)

def download_videos():
    """
    Main entrypoint for downloading the videos of the BRACE-dataset.
    """
    
    print("Downloading videos...")
    
    # Reading csv that contains the names of the videos to download.
    csv = pd.read_csv(OUTPUT_PATH + METAINFO_NAME)
    video_ids = csv["video_id"].to_list()
    
    # Download each video in the loaded csv-file.
    for video_id in tqdm(video_ids, desc="Downloading videos", leave=False):
        _download_video(video_id)
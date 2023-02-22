import os
import zipfile
import wget
import pandas as pd
from pytube import YouTube
from tqdm import tqdm
from global_variables import *
from global_variables import *

def _download_zipfiles():
    """
    Function for downloading the zip-files that contains the keypoints.
    """
    
    keypoint_urls = [
        "https://github.com/dmoltisanti/brace/releases/download/v1.0/dataset.zip",
        "https://github.com/dmoltisanti/brace/releases/download/mk_v1.0/manual_keypoints.zip"
    ]

    for keypoint_url in keypoint_urls:
        wget.download(keypoint_url, out=OUTPUT_PATH + CORPUS_FOLDERS["keypoints_folder"])

def _unzip_keypoints():
    """
    Function for unzipping the downloaded zip-files containing the keypoints.
    """
    
    for i in range(len(DOWNLOAD_FOLDER_NAMES)):
        with zipfile.ZipFile(OUTPUT_PATH + CORPUS_FOLDERS["keypoints_folder"] + DOWNLOAD_FOLDER_NAMES[i] + ".zip", 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_PATH + CORPUS_FOLDERS["keypoints_folder"] + SAVE_FOLDER_NAMES[i] + "/")

def _delete_zipfiles():
    """
    Function for deleting the downloaded zip-files.
    """
    
    for folder_name in DOWNLOAD_FOLDER_NAMES:
        os.remove(OUTPUT_PATH + CORPUS_FOLDERS["keypoints_folder"] + folder_name + ".zip")

def _download_keypoints():    
    """
    Main entrypoint for downloading the keypoints.
    """
    
    # Downloading the zip-files containing the keypoints
    print()
    print("Downloading keypoints...")
    _download_zipfiles()
    
    # Extracting the downloaded zip-files with the keypoints
    print()
    print("Unzipping keypoints...")
    _unzip_keypoints()
    
    # Deleting the zip-files with the keypoints, since we do not need them anymore
    # as they have been extracted
    print()
    print("Deleting keypoint zipfiles...")
    _delete_zipfiles()

def _download_metainfo():
    """
    Function for downloading the meta-information of the videos.
    """
    
    metainfo_url = "https://raw.githubusercontent.com/dmoltisanti/brace/main/videos_info.csv"
    wget.download(metainfo_url, out=OUTPUT_PATH + METAINFO_NAME)

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

def _download_videos():
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
        
def main():
    """
    Main entrypoint for downloading the BRACE-dataset.
    """
    
    _download_keypoints()
    _download_metainfo()
    _download_videos()
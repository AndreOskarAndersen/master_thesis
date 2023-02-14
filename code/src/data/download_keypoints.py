import os
import zipfile
import wget
from global_variables import *

def _download_zipfiles():
    keypoint_urls = [
        "https://github.com/dmoltisanti/brace/releases/download/v1.0/dataset.zip",
        "https://github.com/dmoltisanti/brace/releases/download/mk_v1.0/manual_keypoints.zip"
    ]

    for keypoint_url in keypoint_urls:
        wget.download(keypoint_url, out=OUTPUT_PATH + CORPUS_FOLDERS["keypoints_folder"])

def _unzip_keypoints():
    for i in range(len(DOWNLOAD_FOLDER_NAMES)):
        with zipfile.ZipFile(OUTPUT_PATH + CORPUS_FOLDERS["keypoints_folder"] + DOWNLOAD_FOLDER_NAMES[i] + ".zip", 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_PATH + CORPUS_FOLDERS["keypoints_folder"] + SAVE_FOLDER_NAMES[i] + "/")

def _delete_zipfiles():
    for folder_name in DOWNLOAD_FOLDER_NAMES:
        os.remove(OUTPUT_PATH + CORPUS_FOLDERS["keypoints_folder"] + folder_name + ".zip")

def download_keypoints():    
    print("Downloading keypoints...")
    _download_zipfiles()
    
    print("Unzipping keypoints...")
    _unzip_keypoints()
    
    print("Deleting keypoint zipfiles...")
    _delete_zipfiles()
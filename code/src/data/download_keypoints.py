import os
import zipfile
import wget
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

def download_keypoints():    
    """
    Main entrypoint for downloading the keypoints.
    """
    
    # Downloading the zip-files containing the keypoints
    print("Downloading keypoints...")
    _download_zipfiles()
    
    # Extracting the downloaded zip-files with the keypoints
    print("Unzipping keypoints...")
    _unzip_keypoints()
    
    # Deleting the zip-files with the keypoints, since we do not need them anymore
    # as they have been extracted
    print("Deleting keypoint zipfiles...")
    _delete_zipfiles()
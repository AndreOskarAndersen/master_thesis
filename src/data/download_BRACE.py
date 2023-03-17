import os
import zipfile
import wget
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
        wget.download(keypoint_url, out=OUTPUT_PATH + BRACE_DOWNLOAD + KEYPOINTS_FOLDER)

def _unzip_keypoints():
    """
    Function for unzipping the downloaded zip-files containing the keypoints.
    """
    
    for i in range(len(DOWNLOAD_FOLDER_NAMES)):
        with zipfile.ZipFile(OUTPUT_PATH + BRACE_DOWNLOAD + KEYPOINTS_FOLDER + DOWNLOAD_FOLDER_NAMES[i] + ".zip", 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_PATH + BRACE_DOWNLOAD + KEYPOINTS_FOLDER + SAVE_FOLDER_NAMES[i] + "/")

def _delete_zipfiles():
    """
    Function for deleting the downloaded zip-files.
    """
    
    for folder_name in DOWNLOAD_FOLDER_NAMES:
        os.remove(OUTPUT_PATH + BRACE_DOWNLOAD + KEYPOINTS_FOLDER + folder_name + ".zip")

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
    wget.download(metainfo_url, out=OUTPUT_PATH + BRACE_DOWNLOAD + METAINFO_NAME)
        
def main():
    """
    Main entrypoint for downloading the BRACE-dataset.
    """
    
    _download_keypoints()
    _download_metainfo()
    
if __name__ == "__main__":
    main()
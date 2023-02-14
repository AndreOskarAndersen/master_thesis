import os
from download_keypoints import download_keypoints
from download_metainfo import download_metainfo
from download_videos import download_videos
from global_variables import *

def _make_corpus_folders():
    """
    Function for making various folders for storing the data
    """
    
    # Making overall folder for the unprocessed-data, in case it does not already exist.
    try:
        os.mkdir(OUTPUT_PATH)
    except:
        print(f"Folder {OUTPUT_PATH} already exists. Using existing folder.")
    
    # Making the folders for storing the unprocessed keypoints and videos folders
    # in case they do not already exist
    for corpus_folder in CORPUS_FOLDERS.values():
        try:
            os.mkdir(OUTPUT_PATH + corpus_folder)
        except:
            print(f"Folder {OUTPUT_PATH + corpus_folder} already exists. Using existing folder.")

def main():
    """
    Main entrypoint for downloading the BRACE-dataset.
    """
    
    # Making various folders for storing the data
    _make_corpus_folders()
    
    # Dowloading the data
    download_keypoints()
    download_metainfo()
    download_videos()

if __name__ == "__main__":
    main()
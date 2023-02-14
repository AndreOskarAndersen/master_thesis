import os
from download_keypoints import download_keypoints
from download_metainfo import download_metainfo
from download_videos import download_videos
from global_variables import *

def _make_dir(path):
    """
    Makes a directory if it does not already exist.
    
    Parameters
    ----------
    path : str
        Path of the directory to create.
    """
    
    try:
        os.mkdir(path)
    except:
        print(f"Folder {path} already exists. Using existing folder.")

def _make_corpus_folders():
    """
    Function for making various folders for storing the data
    """
    
    # Making the overall folder for the data, in case it does not extist
    _make_dir(DATA_PATH)
    
    # Making overall folder for the unprocessed-data, in case it does not already exist.
    _make_dir(OUTPUT_PATH)
    
    # Making the folders for storing the unprocessed keypoints and videos folders
    # in case they do not already exist
    for corpus_folder in CORPUS_FOLDERS.values():
        _make_dir(OUTPUT_PATH + corpus_folder)

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
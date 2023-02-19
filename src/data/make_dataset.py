import os
import download_BRACE
import download_penn_action
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
    
    # Dowloading the BRACE data
    download_BRACE.main()
    
    # Downloading the Penn Action data
    download_penn_action.main()

if __name__ == "__main__":
    main()
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
    
    # Making the overall folder for the data, in case it does not exist
    _make_dir(DATA_PATH)
    
    # Making overall folder for the unprocessed-data, in case it does not already exist.
    _make_dir(OUTPUT_PATH)
    
    # Making the BRACE-download folder
    _make_dir(OUTPUT_PATH + BRACE_DOWNLOAD)
    
    # Making the Penn_Action-download folder
    _make_dir(OUTPUT_PATH + PENN_ACTION_EXTRACTED)
    
    # Making BRACE subfolders
    _make_dir(OUTPUT_PATH + BRACE_DOWNLOAD + KEYPOINTS_FOLDER)

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
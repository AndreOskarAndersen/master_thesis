import os
import preprocess_BRACE
import preprocess_penn_action
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
    
    # Making overall folder for the processed-data, in case it does not already exist.
    _make_dir(OVERALL_DATA_FOLDER)
    
    # Making the folders for storing the unprocessed keypoints and videos folders
    # in case they do not already exist
    for sub_folder in SUB_DATA_FOLDERS.values():
        _make_dir(OVERALL_DATA_FOLDER + sub_folder) 
            
if __name__ == "__main__":
    _make_corpus_folders()
    
    # Preprocessing BRACE
    preprocess_BRACE.preprocess()
    
    # Preprocessing Penn Action
    preprocess_penn_action.preprocess()
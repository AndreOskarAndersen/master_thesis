import os
import tarfile
import wget
import shutil
from global_variables import *

def _remove_unnecessary_files():
    """
    Function for deleting files not to be used.
    """
    
    # Deleting files that are not needed
    for file in PENN_ACTION_FILES_TO_DELTE:
        os.remove(OUTPUT_PATH + file)
    
    # Deleting folders that do not contain keypoints
    for folder in PENN_ACTION_DIRS_TO_DELETE:
        shutil.rmtree(OUTPUT_PATH + PENN_ACTION_EXTRACTED + folder)

def _download_dataset():
    """
    Function for downloading the Penn Action dataset.
    """
    
    print("\nDownloading Penn Action...")
    url = "https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz"
    wget.download(url, out=OUTPUT_PATH + PENN_ACTION_DOWNLOAD)
    
def _extract_dataset():
    """
    Function for extracting the downloaded Penn Action dataset.
    """
    
    # Extracting zip-file
    with tarfile.open(OUTPUT_PATH + PENN_ACTION_DOWNLOAD) as tar:
        tar.extractall(path=OUTPUT_PATH + PENN_ACTION_EXTRACTED)
        
    # Removing unnecessary files
    print("\nRemoving unneccessary files...")
    _remove_unnecessary_files()
    
def main():
    """
    Main entrypoint for downloading and extracting the Penn Action dataset.
    """
    
    _download_dataset()
    _extract_dataset()
    
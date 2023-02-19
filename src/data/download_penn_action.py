import os
import tarfile
import wget
from global_variables import *

def _download_dataset():
    """
    Function for downloading the Penn Action dataset.
    """
    
    url = "https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz"
    wget.download(url, out=OUTPUT_PATH + PENN_ACTION_DOWNLOAD)
    
def _extract_dataset():
    """
    Function for extracting the downloaded Penn Action dataset.
    """
    with tarfile.open(OUTPUT_PATH + PENN_ACTION_DOWNLOAD) as tar:
        tar.extractall(path=OUTPUT_PATH + PENN_ACTION_EXTRACTED)
        
    os.remove(OUTPUT_PATH + PENN_ACTION_DOWNLOAD)
    
def main():
    """
    Main entrypoint for downloading and extracting the Penn Action dataset.
    """
    
    _download_dataset()
    _extract_dataset()
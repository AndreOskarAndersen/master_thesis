import wget
from global_variables import *

def download_metainfo():
    """
    Function for downloading the meta-information of the videos.
    """
    
    metainfo_url = "https://raw.githubusercontent.com/dmoltisanti/brace/main/videos_info.csv"
    wget.download(metainfo_url, out=OUTPUT_PATH + METAINFO_NAME)
import wget
from global_variables import *

def download_metainfo():
    metainfo_url = "https://raw.githubusercontent.com/dmoltisanti/brace/main/videos_info.csv"
    wget.download(metainfo_url, out=OUTPUT_PATH + METAINFO_NAME)
# =============== GENERAL VARIABLES ===============

# Path to the overall data folder
DATA_PATH = "../../data/"

# Path to overall folder for unprocessed data
OUTPUT_PATH = DATA_PATH + "raw/" 

# =============== BRACE VARIABLES ===============

# Mapping to names of folders for the unprocessed keypoints and videos
CORPUS_FOLDERS = {"keypoints_folder": "keypoints/", "videos_folder": "videos/"}

# Name of the csv containing the metainformation of the videos
METAINFO_NAME = "metainfo.csv"

# Names of the datasets once downloaded
DOWNLOAD_FOLDER_NAMES = ["dataset", "manual_keypoints"]

# List of names to rename the keypoints once they have been downloaded
SAVE_FOLDER_NAMES = ["automatic_keypoints", "manual_keypoints"]

# =============== PENN ACTION VARIABLES ===============

# Name of downloaded dataset
PENN_ACTION_DOWNLOAD = "penn_action.tar.gz"

# Name of the extracted dataset
PENN_ACTION_EXTRACTED = "penn_action/"
# Path to the overall data folder
DATA_PATH = "../../data/"

# Path to overall folder for unprocessed data
RAW_DATA_FOLDER = DATA_PATH + "raw/" 

# Mapping to names of folders for the unprocessed keypoints and videos
RAW_CORPUS_FOLDERS = {"keypoints_folder": "keypoints/", "videos_folder": "videos/"}

# Name of the csv containing the metainformation of the videos
METAINFO_NAME = "metainfo.csv"

# List of names to rename the datasets once they have been downloaded
RAW_KEYPOINT_FOLDERS = {"automatic": "automatic_keypoints/", "manual": "manual_keypoints/"}

# List of names of subfolders of keypoints
RAW_KEYPOINTS_SUBFOLDERS = {"automatic": "dataset/", "manual": "manual_keypoints/"}

# Path to overall folder for the processed data
OVERALL_DATA_FOLDER = "../../data/processed/"

# Name of folders with data
SUB_DATA_FOLDERS = {"keypoints": "keypoints/", "videos": "videos/"} 
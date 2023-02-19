# =============== GENERAL VARIABLES ===============

# Path to the overall data folder
DATA_PATH = "../../data/"

# Path to overall folder for unprocessed data
RAW_DATA_FOLDER = DATA_PATH + "raw/" 

# Path to overall folder for the processed data
OVERALL_DATA_FOLDER = DATA_PATH + "processed/"

# =============== BRACE VARIABLES ===============

# Mapping to names of folders for the unprocessed keypoints and videos
RAW_CORPUS_FOLDERS = {"keypoints_folder": "keypoints/", "videos_folder": "videos/"}

# Name of the csv containing the metainformation of the videos
METAINFO_NAME = "metainfo.csv"

# List of names to rename the datasets once they have been downloaded
RAW_KEYPOINT_FOLDERS = {"automatic": "automatic_keypoints/", "manual": "manual_keypoints/"}

# List of names of subfolders of keypoints
RAW_KEYPOINTS_SUBFOLDERS = {"automatic": "dataset/", "manual": "manual_keypoints/"}

# Name of folders with data
SUB_DATA_FOLDERS = {"keypoints": "keypoints/", "videos": "videos/"} 

# =============== PENN ACTION VARIABLES ===============

# Path for Penn Action Raw data
PENN_ACTION_RAW_PATH = RAW_DATA_FOLDER + "penn_action/Penn_Action/"

# Penn action sub data folders
SUB_PENN_ACTION_FOLDERS = {"keypoints": "labels/", "frames": "frames/"}

# Set of actions from the Penn Action dataset that are relevant.
RELEVANT_ACTIONS = {"baseball_pitch", "bench_press", "sit_ups"}
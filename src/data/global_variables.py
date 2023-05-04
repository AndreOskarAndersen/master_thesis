# =============== GENERAL VARIABLES ===============

# Path to the overall data folder
DATA_PATH = "../../data/"

# Path to overall folder for unprocessed data
OUTPUT_PATH = DATA_PATH + "raw/" 

# =============== BRACE VARIABLES ===============

BRACE_DOWNLOAD = "BRACE/"

# Mapping to names of folders for the unprocessed keypoints and videos
KEYPOINTS_FOLDER = "keypoints/"

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

PENN_ACTION_SUB_FOLDER = "Penn_Action/"

PENN_ACTION_DIRS_TO_DELETE = [f"{PENN_ACTION_SUB_FOLDER}frames/", f"{PENN_ACTION_SUB_FOLDER}tools/"]

PENN_ACTION_FILES_TO_DELTE = [f"{PENN_ACTION_EXTRACTED}{PENN_ACTION_SUB_FOLDER}README", PENN_ACTION_DOWNLOAD]

# =============== CLIMBALONG VARIABLES ===============

# Path to the overall ClimbAlong raw-data
CLIMBALONG_OVERALL_FOLDER = OUTPUT_PATH + "ClimbAlong/"

# Names of the raw ClimbALong-subfolders
CLIMBALONG_SUB_FOLDERS  = {"x": "input/", "y": "target/", "video": "video/"}
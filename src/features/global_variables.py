# =============== GENERAL VARIABLES ===============

# Path to the overall data folder
DATA_PATH = "../../data/"

# Path to overall folder for unprocessed data
RAW_DATA_FOLDER = DATA_PATH + "raw/" 

# Path to overall folder for the processed data
OVERALL_DATA_FOLDER = DATA_PATH + "processed/"

# Wanted height and width
TARGET_HEIGHT, TARGET_WIDTH = 50, 50

# Number of keypoints to use
NUM_KEYPOINTS = 25

CLIMBALONG_KEYPOINTS = {"nose": 0, "left_ear": 1, "right_ear": 2, "left_shoulder": 3, "right_shoulder": 4, "left_elbow": 5, "right_elbow": 6, "left_wrist": 7, "right_wrist": 8, "left_pinky": 9, "right_pinky": 10, "left_index": 11, "right_index": 12, "left_thumb": 13, "right_thumb": 14, "left_hip": 15, "right_hip": 16, "left_knee": 17, "right_knee": 18, "left_ankle": 19, "right_ankle": 20, "left_heel": 21, "right_heel": 22, "left_foot_index": 23, "right_foot_index": 24}

# =============== BRACE VARIABLES ===============

# Name of the csv containing the metainformation of the videos
METAINFO_NAME = "metainfo.csv"

RAW_BRACE_PATH = RAW_DATA_FOLDER + "BRACE/"

# List of names to rename the datasets once they have been downloaded
RAW_KEYPOINT_FOLDERS = {"automatic": "keypoints/automatic_keypoints/", "manual": "keypoints/manual_keypoints/"}

# List of names of subfolders of keypoints
RAW_KEYPOINTS_SUBFOLDERS = {"automatic": "dataset/", "manual": "manual_keypoints/"}

# Original height and width of BRACE videos
BRACE_HEIGHT, BRACE_WIDTH = 1080, 1920

# Description of keypoint-indices
BRACE_KEYPOINTS = {0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}

# =============== PENN ACTION VARIABLES ===============

# Path for Penn Action Raw data
PENN_ACTION_RAW_PATH = RAW_DATA_FOLDER + "penn_action/Penn_Action/"

# Penn action raw keypoints path 
PENN_ACTION_RAW_KEYPOINTS_PATH = PENN_ACTION_RAW_PATH + "labels/"

# Set of actions from the Penn Action dataset that are relevant.
RELEVANT_ACTIONS = {"baseball_pitch", "bench_press", "sit_ups"}

# Description of keypoint.indices
PENN_ACTION_KEYPOINTS = {0: "nose", 1: "left_shoulder", 2: "right_shoulder", 3: "left_elbow", 4: "right_elbow", 5: "left_wrist", 6: "right_wrist", 7: "left_hip", 8: "right_hip", 9: "left_knee", 10: "right_knee", 11: "left_ankle", 12: "right_ankle"}
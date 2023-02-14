import os
from download_keypoints import download_keypoints
from download_metainfo import download_metainfo
from download_videos import download_videos
from global_variables import *

def make_corpus_folders():
    try:
        os.mkdir(OUTPUT_PATH)
    except:
        print(f"Folder {OUTPUT_PATH} already exists. Using existing folder.")
    
    for corpus_folder in CORPUS_FOLDERS.values():
        try:
            os.mkdir(OUTPUT_PATH + corpus_folder)
        except:
            print(f"Folder {OUTPUT_PATH + corpus_folder} already exists. Using existing folder.")

def main():
    #make_corpus_folders()
    #download_keypoints()
    #download_metainfo()
    download_videos()

if __name__ == "__main__":
    main()
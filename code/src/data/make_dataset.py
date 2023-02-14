import os
import zipfile
import wget

OUTPUT_PATH = "../../raw/keypoints"
FOLDER_NAMES = ["dataset", "manual_keypoints"]

def download_keypoints():
    keypoint_urls = [
        "https://github.com/dmoltisanti/brace/releases/download/v1.0/dataset.zip",
        "https://github.com/dmoltisanti/brace/releases/download/mk_v1.0/manual_keypoints.zip"
    ]

    for keypoint_url in keypoint_urls:
        wget.download(keypoint_url, out=OUTPUT_PATH)

def unzip_keypoints():
    for folder_name in FOLDER_NAMES:
        with zipfile.ZipFile(OUTPUT_PATH + folder_name + ".zip", 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_PATH + folder_name + "/")

def delete_keypoint_zipfiles():
    for folder_name in FOLDER_NAMES:
        os.remove(OUTPUT_PATH + folder_name + ".zip")

def main():
    print("Downloading keypoints...")
    download_keypoints()
    
    print("Unzipping keypoints...")
    unzip_keypoints()
    
    print("Deleting keypoint zipfiles...")
    delete_keypoint_zipfiles()

if __name__ == "__main__":
    main()
##
import numpy as np
import requests
import argparse
import tarfile
import shutil
import glob
import cv2
import os

##
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download CALTECT256 dataset.")
    parser.add_argument("--destination", default='./data/', help="Path to save the dataset.")
    parser.add_argument("--filename", default="256_ObjectCategories.tar", help="Filename to save file.")
    parser.add_argument("--dirname", default="256_ObjectCategories", help="Dirname after extraction")
    parser.add_argument("--url", default="http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar", help="URL")
    return parser.parse_args()


args = parse_arguments()
tar_ball = os.path.join(args.destination, args.filename)
src_data = os.path.join(args.destination, args.dirname)

## DOWNLOAD
##
def download_dataset():
    # Check whether the directories exist
    if not os.path.isdir(args.destination):     # Check destination dir.
        os.makedirs(args.destination)           # If not exist, create dir.

    # Check if the file already exist
    if os.path.exists(tar_ball):
        print(">> Tar ball already exists. Skip downloading.")
    else:
        print(">> Downloading file... ")
        r = requests.get(args.url)
        with open(tar_ball, "wb") as code:
            code.write(r.content)
        print("   Done!")

##
def extract_dataset():
    ##
    # Extract the tarball
    if os.path.exists(src_data):
        print(">> Tarball already extracted. Skip extracting")
    else:
        print(">> Extracting tarball")
        if (tar_ball.endswith("tar.gz")):
            tar = tarfile.open(tar_ball, "r:gz")
            tar.extractall(path=args.destination)
            tar.close()
        elif (tar_ball.endswith("tar")):
            tar = tarfile.open(tar_ball, "r:")
            tar.extractall(path=args.destination)
            tar.close()
        print("   Done!")

def cleanup():
    print("   Cleanup")
    os.remove(tar_ball)
    print("   Done!")

#
if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    cleanup()
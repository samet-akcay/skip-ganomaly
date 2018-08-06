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
    parser = argparse.ArgumentParser(description="Sliding Window")
    parser.add_argument("--destination", default='./data/', help="Path to save the dataset.")
    parser.add_argument("--filename", default="UCSD_Anomaly_Dataset.tar.gz", help="Filename to save file.")
    parser.add_argument("--dirname", default="UCSD_Anomaly_Dataset.v1p2", help="Dirname after extraction")
    parser.add_argument("--url", default="http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz", help="URL")
    parser.add_argument("--gt_filename", default="UCSD_Ped1.tar.gz", help="Default filename for pixel gt for ped1 dataset")
    parser.add_argument("--gt_dirname", default="UCSD_Ped1", help="GT Dirname after extraction")    
    parser.add_argument("--gt_url", default="https://hciweb.iwr.uni-heidelberg.de/sites/default/files/node/files/1109511602/UCSD_Ped1.tar.gz", help="GT URL")
    return parser.parse_args()


args = parse_arguments()
tar_ball = os.path.join(args.destination, args.filename)
src_data = os.path.join(args.destination, args.dirname)
gt_tar_ball = os.path.join(args.destination, args.gt_filename)
gt_src_data = os.path.join(args.destination, args.gt_dirname)


##
def download_dataset():
    ##
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

    # Check if the gt file already exist
    if os.path.exists(gt_tar_ball):
        print(">> GT Tar ball already exists. Skip downloading.")
    else:
        print(">> Downloading GT file... ")
        r = requests.get(args.gt_url)
        with open(gt_tar_ball, "wb") as code:
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

    # Extract the GT tarball
    if os.path.exists(gt_src_data):
        print(">> Tarball already extracted. Skip extracting")
    else:
        print(">> Extracting tarball")
        if (gt_tar_ball.endswith("tar.gz")):
            tar = tarfile.open(gt_tar_ball, "r:gz")
            tar.extractall(path=args.destination)
            tar.close()
        elif (gt_tar_ball.endswith("tar")):
            tar = tarfile.open(gt_tar_ball, "r:")
            tar.extractall(path=args.destination)
            tar.close()
        print("   Done!")

##
def restructure_dataset():
    ##
    # Restructure dataset.
    tar_data = os.path.join(args.destination, "UCSD_Anomaly_Dataset")

    if not os.path.exists(tar_data):
        os.makedirs(tar_data)

    dsets = [d for d in sorted(os.listdir(src_data)) if os.path.isdir(os.path.join(src_data, d))]
    splits = ["Train", "Test"]
    classes = ['nrm', 'abn']
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    print(">> Restructuring the dataset.")

    # Move UCSD Ped1 Pixel GT first
    [shutil.move(os.path.join(gt_src_data, i), os.path.join(src_data,'UCSDped1', 'Test', i)) for i in os.listdir(gt_src_data)]

    for dset in dsets:
        for split in splits:
            for root in glob.glob(os.path.join(src_data, dset, split, "*")):
                if os.path.isdir(root):
                    for filename in os.listdir(root):
                        if filename.endswith((".jpg", ".tiff", ".png", ".tif", ".bmp")):
                            dst = os.path.join(tar_data, dset, split.lower(), 'nrm')
                            if  not os.path.exists(dst):
                                os.makedirs(dst)
                            if root.endswith("_gt"):
                                dst_filename = filename.replace("frame", "") if "frame" in filename else filename
                                shutil.move(os.path.join(root, filename), os.path.join(dst, os.path.basename(root[:-3]) + "_" + dst_filename))
                            else:
                                shutil.move(os.path.join(root, filename), os.path.join(dst, os.path.basename(root) + "_" + filename))

    # Split test-set into two classes: normal vs abnormal.
    print("   Extracting abnormal samples from ground truth")
    gt = {
        'UCSDped1': {
            'Test001': list(range(60, 153)),
            'Test002': list(range(50, 176)),
            'Test003': list(range(91, 201)),
            'Test004': list(range(31, 169)),
            'Test005': [i for j in (range(5,91), range(140,201)) for i in j],
            'Test006': [i for j in (range(1,101), range(110,201)) for i in j],
            'Test007': list(range(1,  176)),
            'Test008': list(range(1,  95)),
            'Test009': list(range(1,  49)),
            'Test010': list(range(1,  141)),
            'Test011': list(range(70, 166)),
            'Test012': list(range(130,201)),
            'Test013': list(range(1,157)),
            'Test014': list(range(1,201)),
            'Test015': list(range(138,201)),
            'Test016': list(range(123,201)),
            'Test017': list(range(1,48)),
            'Test018': list(range(54,121)),
            'Test019': list(range(64,139)),
            'Test020': list(range(45,176)),
            'Test021': list(range(31,201)),
            'Test022': list(range(16,108)),
            'Test023': list(range(8,166)),
            'Test024': list(range(50,172)),
            'Test025': list(range(40,136)),
            'Test026': list(range(77,145)),
            'Test027': list(range(10,123)),
            'Test028': list(range(105,201)),
            'Test029': [i for j in (range(1,16), range(45, 114)) for i in j],
            'Test030': list(range(175,201)),
            'Test031': list(range(1,181)),
            'Test032': [i for j in (range(1,53), range(65, 116)) for i in j],
            'Test033': list(range(5,166)),
            'Test034': list(range(1,122)),
            'Test035': list(range(86,201)),
            'Test036': list(range(15,109))
        },
        'UCSDped2': {
            'Test001': list(range(61, 181)),
            'Test002': list(range(95, 181)),
            'Test003': list(range(1,  147)),
            'Test004': list(range(31, 181)),
            'Test005': list(range(1,  130)),
            'Test006': list(range(1,  160)),
            'Test007': list(range(46, 181)),
            'Test008': list(range(1,  181)),
            'Test009': list(range(1,  121)),
            'Test010': list(range(1,  151)),
            'Test011': list(range(1,  181)),
            'Test012': list(range(88, 181))
        }
    }

    for dset in dsets:
        # Create abnormal class for the test set.
        abnormal_dir = os.path.join(tar_data, dset, "test", "abn")
        normal_dir   = os.path.join(tar_data, dset, "test", "nrm")

        if not os.path.exists(abnormal_dir):
            os.makedirs(abnormal_dir)

        for key, values in gt[dset].items():
            for value in values:
                for ext in [".tif", ".bmp"]:
                    if os.path.exists(os.path.join(normal_dir, key + "_" + str(value).zfill(3) + ext)):
                        # Read GT. If no pixel in GT, then move to normal dir.
                        img = cv2.imread(os.path.join(normal_dir, key + "_" + str(value).zfill(3) + ".bmp"))
                        if np.count_nonzero(img) != 0:
                            shutil.move(os.path.join(normal_dir, key + "_" + str(value).zfill(3) + ext),
                                        os.path.join(abnormal_dir, key + "_" + str(value).zfill(3) + ext))

    print("   Cleanup")
    shutil.rmtree(src_data)
    os.remove(tar_ball)
    shutil.rmtree(gt_src_data)
    os.remove(gt_tar_ball)
    print("   Done!")

#
if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    restructure_dataset()

###
# To download the data, run the following command:
#  python3 spair.py --download
###
import os
from typing import Text
import glob
import argparse
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as matpatches
import json
import matplotlib.colors as mcolors


def download_dataset(dataset_dir: Text):
    """Download the dataset to the specified directory."""
    import urllib.request
    import tarfile

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Download the dataset
    url = 'http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz'
    filename = url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(filepath):
        print('Downloading ' + url + ' to ' + filepath)
        urllib.request.urlretrieve(url, filepath)

    # Extract the dataset
    cwd = os.getcwd()
    tar = tarfile.open(filepath)
    os.chdir(dataset_dir)
    tar.extractall()
    tar.close()
    os.chdir(cwd)
    print('Done!')


def iterate_over_pairs(dataset_dir: Text, split: Text = 'test'):
    """Yields the pairs of images and corresponding ground truth points. 
    """
    assert split in ['trn', 'test', 'val']
    for annotation in glob.glob(os.path.join(dataset_dir, 'PairAnnotation', split, '*.json')):
        with open(annotation) as f:
            data = json.load(f)
        category = data['category']
        source_path = os.path.join(
            dataset_dir, 'JPEGImages', category, data['src_imname'])
        target_path = os.path.join(
            dataset_dir, 'JPEGImages', category, data['trg_imname'])
        target_points = data['trg_kps']
        source_points = data['src_kps']
        target_pose = data['trg_pose']
        source_pose = data['src_pose']
        target_bounding_box = data['trg_bndbox']
        source_bounding_box = data['src_bndbox']
        result = {
            'source_path': source_path,
            'target_path': target_path,
            'source_points': source_points,
            'target_points': target_points,
            'category': category,
            'source_pose': source_pose,
            'target_pose': target_pose,
            "mirror": data['mirror'],
            "viewpoint_variation": data['viewpoint_variation'],
            "target_bounding_box": target_bounding_box,
            "source_bounding_box": source_bounding_box
        }
        yield result


def plot_pair_correspondence(source_image: np.ndarray, target_image: np.ndarray,
                             source_points: List[Tuple[int, int]],
                             target_points: List[Tuple[int, int]], patch_size: int = 4,
                             source_bounding_box: List[int] = None, target_bounding_box: List[int] = None):
    """Plots the source and target images, with the corresponding points. 
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 2, 1)  # last value is the current plot
    assert len(source_points) == len(
        target_points), "source and target points must have the same length."
    ax.imshow(source_image)
    colors = np.random.choice(
        list(mcolors.CSS4_COLORS.keys()), len(source_points))
    for (w, h), c in zip(source_points, colors):
        rect = matpatches.Rectangle((w, h), patch_size, patch_size,
                                    linewidth=patch_size // 2, edgecolor=c, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    if source_bounding_box is not None:
        left, top, right, bottom = source_bounding_box
        rect = matpatches.Rectangle((left, top), right-left, bottom-top,
                                    linewidth=patch_size // 2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(target_image)
    for (w, h), c in zip(target_points, colors):
        rect = matpatches.Rectangle((w, h), patch_size, patch_size,
                                    linewidth=patch_size // 2, edgecolor=c, facecolor='none')
        ax.add_patch(rect)
    if target_bounding_box is not None:
        left, top, right, bottom = target_bounding_box
        rect = matpatches.Rectangle((left, top), right-left, bottom-top,
                                    linewidth=patch_size // 2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()


def pck(target_image: np.ndarray, target_points: List[Tuple[int, int]], calculated_points: List[Tuple[int, int]], threshold: float):
    """Calculates the PCK metric, given a threshold."""
    assert len(target_points) == len(
        calculated_points), "target and calculated points must have the same length."
    max_size = max(target_image.shape[0], target_image.shape[1])
    pck_metric = 0
    for (w, h), (x, y) in zip(target_points, calculated_points):
        pck_metric += np.sqrt((w - x) ** 2 + (h - y) **
                              2) <= threshold * max_size
    return pck_metric, len(target_points)


def bounding_box_pck(target_bounding_box: List[int], target_points: List[Tuple[int, int]], calculated_points: List[Tuple[int, int]], threshold: float):
    """Calculates the PCK metric, given a threshold and a bounding box."""
    assert len(target_points) == len(
        calculated_points), "target and calculated points must have the same length."
    left, top, right, bottom = target_bounding_box
    max_size = max(right-left, bottom-top)
    pck_metric = 0
    for (w, h), (x, y) in zip(target_points, calculated_points):
        pck_metric += np.sqrt((w - x) ** 2 + (h - y) **
                              2) <= threshold * max_size
    return pck_metric, len(target_points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', dest='download', action='store_true',
                        help='Downlad the spair dataset and extract it.')
    parser.set_defaults(download=False)
    parser.add_argument('--dataset_dir', type=str, dest='dataset_dir', default='.',
                        help='Directory where the dataset is stored.')
    args = parser.parse_args()

    if args.download:
        download_dataset(args.dataset_dir)

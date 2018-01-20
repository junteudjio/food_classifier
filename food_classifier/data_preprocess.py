import logging
import argparse
import os
import glob
import shutil
import random
import zipfile

from tqdm import tqdm
from six.moves.urllib.request import urlretrieve

import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('./logs/data_preprocess.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

__author__ = 'Junior Teudjio'
__all__ = ['maybe_download']


raw_dataset_base_url = 'http://research.us-east-1.s3.amazonaws.com/public/'
raw_filesize_expected = None
raw_filename = 'sushi_or_sandwich_photos.zip'
extracted_raw_filename = 'sushi_or_sandwich'

def _setup_args():
    parser = argparse.ArgumentParser()
    download_prefix = './downloads'
    data_prefix = './data'
    utils.mkdir_p(download_prefix)
    utils.mkdir_p(data_prefix)
    utils.remove_childreen(data_prefix)

    parser.add_argument("--download_prefix", default=download_prefix, help='Folder where to download raw dataset')
    parser.add_argument("--data_prefix", default=data_prefix, help='Folder where to dump preprocess dataset')

    parser.add_argument("--train_percentage", default=0.85, type=float)
    parser.add_argument("--val_percentage", default=0.10, type=float)
    parser.add_argument("--test_percentage", default=0.05, type=float)
    parser.add_argument("--seed", default=32, type=int, help='Random seed value')

    parser.add_argument("--shuffle_during_split", default=True, action='store_true')
    return parser.parse_args()


def _reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks just transferred [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def maybe_download(url, filename, prefix, num_bytes=None):
    """
    Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename
    num_bytes=None disables the file size check.

    Parameters
    ----------
    url : str
        The url of the dataset.
    filename : str
        Filename of the raw downloaded dataset.
    prefix : str
        Parent path where to put the downloaded file
    num_bytes : Long
        Excepted number of bytes of the file for sanity check (optional)

    Returns
    -------
        local_filename: str
    """
    local_filename = None
    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            logger.info("Downloading file {}...".format(url + filename))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, os.path.join(prefix, filename),
                                                reporthook=_reporthook(t))
        except AttributeError as e:
            logger.error("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix, filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        logger.info("File {} successfully loaded".format(filename))
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename


def _copy(class_dst_path, src_img_filepaths):
    '''
    Copy a list of list of image files to a destination folder.

    Parameters
    ----------
    class_dst_path : str
        Destination folder where to copy the images.
    src_img_filepaths: list
        List of images filepaths to be copied.

    Returns
    -------

    '''
    utils.mkdir_p(class_dst_path)
    for src_img_path in src_img_filepaths:
        img_filename = src_img_path.split('/')[-1]
        dst_img_path = os.path.join(class_dst_path, img_filename)
        shutil.copyfile(src_img_path, dst_img_path)

def _split_tier(raw_dataset_path, data_prefix, percents, shuffle=False, seed=32):
    '''
    Split the downloaded dataset into : train | validation | test sets.

    Parameters
    ----------
    raw_dataset_path : str
        Path where to find the original dataset.
    data_prefix : str
        Parent folder of train, validation and test folders.
    percents: dict (eg: {train: 0.85, val: 0.10, test: 0.05}
        Percentages values of each spliting tier (should sum to 1.0).
    shuffle: boolean
        Determine whether to shuffle the dataset before splitting
    seed: int
        Seed value used during random splitting

    Returns
    -------

    '''
    random.seed(seed)
    total_percentages = percents['train'] + percents['val'] + percents['test']
    assert 0.0 <= (1.0 - total_percentages) <= 1e-3, 'spliting percentages must sum to 1.0'

    # create paths where to save the tiers
    train_path = os.path.join(data_prefix, 'train')
    val_path = os.path.join(data_prefix, 'validation')
    test_path = os.path.join(data_prefix, 'test')
    utils.mkdir_p(train_path)
    utils.mkdir_p(val_path)
    utils.mkdir_p(test_path)

    # get the classes directories
    classes_names = [dir_name for dir_name in os.listdir(raw_dataset_path)
                    if dir_name!='.DS_Store']
    logger.info('Images classes in the datasets are: {}'.format(str(classes_names)))
    for class_name in classes_names:
        class_path = os.path.join(raw_dataset_path, class_name)
        all_imgs = glob.glob(class_path + '/train_*')
        if shuffle:
           random.shuffle(all_imgs)

        # split the images for this class into 3 tiers
        train_imgs = all_imgs[:int(len(all_imgs)*percents['train'])]
        other_imgs = all_imgs[int(len(all_imgs)*percents['train']):]
        val_imgs = other_imgs[:int(len(all_imgs)*percents['val'])]
        test_imgs = other_imgs[int(len(all_imgs)*percents['val']):]

        # saving folders
        class_dst_path_train = os.path.join(train_path, class_name)
        class_dst_path_val = os.path.join(val_path, class_name)
        class_dst_path_test = os.path.join(test_path, class_name)

        # move each tier images for this class in right the dst path
        _copy(class_dst_path_train, train_imgs)
        _copy(class_dst_path_val, val_imgs)
        _copy(class_dst_path_test, test_imgs)


def main():
    args = _setup_args()

    download_prefix = args.download_prefix
    data_prefix = args.data_prefix
    percents = {
        'train': args.train_percentage,
        'val': args.val_percentage,
        'test': args.test_percentage,
    }

    logger.info("Downloading dataset into {}".format(download_prefix))
    # Download and extract the raw dataset
    raw_data_zip = maybe_download(raw_dataset_base_url, raw_filename, download_prefix, raw_filesize_expected)
    raw_data_zip_ref = zipfile.ZipFile(os.path.join(download_prefix, raw_filename), 'r')
    raw_data_zip_ref.extractall(download_prefix)
    raw_data_zip_ref.close()
    logger.info('Downloading dataset OK!')

    logger.info("Preprocessing dataset into {}".format(data_prefix))
    # Split train into train, validation and test into 85-10-5
    # Shuffle train, validation, test
    logger.info("Splitting the dataset into train, validation and test")
    raw_dataset_path = os.path.join(download_prefix, extracted_raw_filename)
    _split_tier(raw_dataset_path, data_prefix, percents, shuffle=args.shuffle_during_split, seed=args.seed)
    logger.info('Preprocessing dataset OK!')


if __name__ == '__main__':
    main()

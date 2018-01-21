import errno
import os
import shutil

__author__ = 'Junior Teudjio'
__all__ = ['mkdir_p', 'remove_childreen']


def mkdir_p(path):
    '''
    Recursively creates the directories in a given path
    Equivalent to batch cmd mkdir -p.

    Parameters
    ----------
    path : str
        Path to the final directory to create.
    Returns
    -------

    '''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def remove_childreen(path):
    '''
    Delete all the children folders/files in a directory.

    Parameters
    ----------
    path : str
       Path to the directory to clean up.
    Returns
    -------

    '''
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
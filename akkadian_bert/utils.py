import errno
import os
from argparse import ArgumentTypeError
from typing import Tuple

WINDOWS_SIZE = 20


def create_dirs_for_file(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def get_concatenated_files(file1, file2, encoding='utf-8'):
    with open(file1, "r", encoding=encoding) as f_1:
        data = f_1.readlines()
    with open(file2, "r", encoding=encoding) as f_2:
        data.extend(f_2.readlines())
    return data


def calc_wind_around_ind(ind: int, upper_bound: int, window_size: int = WINDOWS_SIZE) -> Tuple[int, int]:
    """
    This function calculates a window around a given ind of size at most window_size * 2.
    The window is bounded by 0 from below and by a given upper bound.

    :param ind: A given index to build the window around
    :param upper_bound: A given upper bound to the window
    :param window_size: a given window_size to one direction
    :return: two integers representing the lowest and largest indices of the window.
    """
    min_index = max(0, ind - window_size)
    max_index = min(min_index + window_size * 2, upper_bound)
    min_index = max(0, max_index - window_size * 2)
    return min_index, max_index


def natural_number(n):
    if not is_natural_number(n):
        raise ArgumentTypeError(f"Given {n} is not a natural number")
    return int(n)


def is_natural_number(n):
    try:
        num = int(n)
    except ValueError:
        return False
    return num > 0


def text_language(lang: str):
    if lang != "Akkadian" and lang != "English":
        raise ArgumentTypeError(f"Given {lang} is not an optional language of ORACC texts")
    return lang

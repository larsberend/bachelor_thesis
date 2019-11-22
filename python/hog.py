from typing import Sequence, Iterator, Tuple, Optional, Union, Any

import numpy as np
import math



def hog(
    angles: Sequence[float],
    magnitude: Sequence[float],
    num_bins: int,
    max_angle: int = 360,
    normalize_threshold: Optional[float] = None,
    normalize_mean: bool = False,
):
    """
    Create a histogram of orientated gradients.
    @author: Christopher Gundler
    >>> hog([80], [2], num_bins=9, max_angle=180)
    array([0., 0., 0., 0., 2., 0., 0., 0., 0.], dtype=float32)
    >>> hog([180], [2], num_bins=9, max_angle=180)
    array([2., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> hog([10], [4], num_bins=9, max_angle=180)
    array([2., 2., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> hog([165], [85], num_bins=9, max_angle=180)
    array([21.25,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 63.75],
          dtype=float32)

    :param angles:
    :param magnitude:
    :param num_bins:
    :param max_angle:
    :return:
    """
    bins = np.zeros((num_bins,), dtype=np.float32)
    bin_width = max_angle / num_bins
    for i, angle in enumerate(angles):
        left_bin_index = int(angle // bin_width)
        split = 1 - ((angle / bin_width) - left_bin_index)
        if math.isclose(split, 1.0):
            bins[left_bin_index % num_bins] += magnitude[i]
        else:
            bins[left_bin_index % num_bins] += magnitude[i] * split
            bins[(left_bin_index + 1) % num_bins] += magnitude[i] * (1 - split)

    if normalize_threshold is not None:
        num_elements = np.count_nonzero(np.asarray(magnitude) >= normalize_threshold)
        if num_elements > 0:
            bins /= num_elements

    if normalize_mean:
        bins -= bins.mean()
        bins[bins < 0] = 0

    return bins


def segment(data: Sequence, return_value: bool = False) -> Iterator[Union[Tuple[int, int], Tuple[int, int, Any]]]:
    """
    Segment an sequence of values.
    >>> tuple(segment([True, True, False, True, False, False]))
    ((0, 1), (2, 2), (3, 3), (4, 5))
    >>> tuple(segment([True, False, False, True, True, False, True]))
    ((0, 0), (1, 2), (3, 4), (5, 5), (6, 6))
    >>> tuple(segment([True, True, False, True, False, False], True))
    ((0, 1, True), (2, 2, False), (3, 3, True), (4, 5, False))

    :param data: The sequence of elements
    :return: Segments of data
    """
    start_index = 0
    for i, value in enumerate(data):
        if value != data[start_index]:
            yield (start_index, i - 1) if not return_value else (start_index, i - 1, data[start_index])
            start_index = i
    yield (start_index, len(data) - 1) if not return_value else (start_index, len(data) - 1, data[start_index])

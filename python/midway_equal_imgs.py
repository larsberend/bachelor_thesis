'''
Uses midway equalization to deflicker images in a folder, as described here:
https://hal.archives-ouvertes.fr/hal-00407796v2
'''

import numpy as np
import cv2 as cv
import video
import matplotlib.pyplot as plt

number_of_frames = 2
path = '/home/laars/uni/BA/code/python/results/wflicker/'  # '/home/laars/uni/BA/eyes/000/eye1.mp4'


def midway_equal_imgs(imgs,
                      nr_frames=number_of_frames,
                      save_path='/home/laars/uni/BA/code/python/results/deflicker/'
                      ):
    width = imgs[0].shape[1]
    height = imgs[0].shape[0]
    sum_sort = np.zeros((width * height), dtype=np.uint16)
    indices = np.zeros((nr_frames, width * height), dtype=np.uint32)
    frames_def = np.zeros((nr_frames, height, width), dtype=np.uint8)

    for x in range(nr_frames):
        flat = np.concatenate(imgs[x])
        indices[x] = flat.argsort().argsort()
        sort = np.sort(flat)
        sum_sort = sum_sort + sort
    mean = np.array(sum_sort / nr_frames, dtype=np.uint8)

    for y in range(nr_frames):
        backsort = mean[indices[y]]
        frames_def[y] = np.reshape(backsort, (height, width))

        if save_path is not None:
            cv.imwrite('%s%s.png' % (save_path, y),
                       frames_def[y]
                       )

    return frames_def


if __name__ == '__main__':
    imgs = []
    for z in range(number_of_frames):
        path_file = path + 'flicker%s.png' % (z)
        imgs.append(cv.imread(path_file, 0))
    x = midway_equal_imgs(imgs)
    print(x)

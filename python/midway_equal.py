'''
Uses midway equalization to deflicker x frames, as described here:
https://hal.archives-ouvertes.fr/hal-00407796v2
'''

import numpy as np
import cv2 as cv
import video
import matplotlib.pyplot as plt

number_of_frames = 1
video_path = '/home/laars/uni/BA/eyes/Gold_Standard_Kopien/1/1d38e5a6-7b4b-49cd-b23c-8dc92b35c265/1d38e5a6-7b4b-49cd-b23c-8dc92b35c265/000/eye1.mp4'


# '/home/laars/uni/BA/eyes/000/eye1.mp4'


def midway_equal(path=video_path,
                 nr_frames=number_of_frames,
                 save_path='/home/laars/uni/BA/code/python/results/deflicker/'
                 ):
    # frames = []

    cap = video.create_capture(path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    sum_sort = np.zeros((width * height))
    indices = []  # np.zeros((nr_frames, width*height),dtype=np.uint16)
    frames_def = np.zeros((nr_frames, height, width))

    for x in range(nr_frames):
        _ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flat = np.concatenate(frame)
        indices.append(flat.argsort().argsort())
        sort = np.sort(flat)
        sum_sort = sum_sort + sort
    mean = np.array(sum_sort / nr_frames, dtype=np.uint8)

    for y in range(nr_frames):
        backsort = mean[indices[y]]
        print(indices)
        frames_def[y] = np.reshape(backsort, (height, width))

        if save_path is not None:
            cv.imwrite('%s%s.png' % (save_path, y + 1),
                       frames_def[y]
                       )

    return frames_def


if __name__ == '__main__':
    midway_equal()

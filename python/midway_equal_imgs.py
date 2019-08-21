'''
Uses midway equalization to deflicker x frames, as described here:
https://hal.archives-ouvertes.fr/hal-00407796v2
'''

import numpy as np
import cv2 as cv
import video
import matplotlib.pyplot as plt

nr_frames = 2
path = '/home/laars/uni/BA/code/python/results/wflicker/'#'/home/laars/uni/BA/eyes/000/eye1.mp4'


def midway_equal_imgs(   path = path,
                    nr_frames = nr_frames,
                    save_path='/home/laars/uni/BA/code/python/results/deflicker/'
                ):
    imgs = []
    for z in range(nr_frames):
        path_file = path + 'flicker%s.png' % (z)
        imgs.append(cv.imread(path_file, 0))
    width = imgs[0].shape[1]
    height = imgs[0].shape[0]
    sum_sort = np.zeros((width * height),dtype=np.uint16)
    indices = np.zeros((nr_frames, width*height),dtype=np.uint32)
    frames_def = np.zeros((nr_frames, height, width), dtype=np.uint8)

    for x in range(nr_frames):

        cv.imshow('frame', imgs[x])
        cv.waitKey(0)

        # print(np.unique(frame, return_counts=True))
        # frames.append(frame)
        flat = np.concatenate(imgs[x])
        indices[x] = flat.argsort().argsort()
        # print(np.unique(np.sort(flat), return_counts=True))
        sort = np.sort(flat)
        # hist = cv.calcHist(frame, [0], None,[256],[0,256])
        # plt.plot(sort)
        # plt.show()
        sum_sort = sum_sort + sort
    mean = np.array(sum_sort/nr_frames, dtype=np.uint8)


    for y in range(nr_frames):
        # print(mean.shape)
        # print(indices)

        backsort = mean[indices[y]]
        frames_def[y] = np.reshape(backsort, (height, width))
        # print(np.unique(frames_def, return_counts=True))
        # hist = cv.calcHist(frame_def, [0], None,[256],[0,256])
        # plt.plot(hist)
        # plt.show()


        cv.imshow('frame_def', frames_def[y])
        cv.waitKey(0)
        if save_path is not None:
            cv.imwrite('%s%s.png' % (save_path, y),
                        frames_def[y]
                        )

    return frames_def

if __name__== '__main__':
    midway_equal_imgs()

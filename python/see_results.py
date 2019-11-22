#!/usr/bin/env python
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import sys
import bz2

# extract, print and show images of .pickle files created by testing.


def show_lifespan(data):
    for ls, det_para, lk_para in zip(data.head(5)['avg_lifespan'][:], data.head(5)['detector_params'][:], data.head(5)['lk_params']):
        print(ls)
        print(det_para)
        print(lk_para)
    # for ls, para in zip(data.head(5)['avg_lifespan'][:], data.head(5)['detector_params'][:]):
    #     print(ls)
    #     print(para)


def show_heat(data):
    for heat, pa in zip(data.head(5)['heatmap'], data.head(5)['detector_params']):
        # print(heat)
        # heatmap_norm = cv.normalize(heat, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                    # dtype=cv.CV_8U).transpose(1, 0)
        cv.imshow('heat.png', heat)
        print(pa)

        # cv.imwrite('heat_not_inv.png', heat.transpose(1,0)) #cv.applyColorMap(heatmap_norm,  cv.COLORMAP_PARULA))
        # break
        # cv.imwrite('heat.png', cv.applyColorMap(heatmap_norm,  cv.COLORMAP_PARULA))


def show_hist(data):
    for hist, pa in zip(data.head(1)['hist_lifespan'], data.head(1)['detector_params']):
        print(hist.shape)
        print(pa)
        # plt.hist(hist ,bins=int(max(hist) / 10))
        # plt.xlabel('time in frames')
        # plt.ylabel('Nr. of tracks')
        # plt.show()


if __name__=='__main__':
    # for i in range(10):
    #     for j in range(2):
    #         print((i,j))
    #         file = bz2.open('results/no_blinks/dataframes/lk/%s_eye%s.mp4_test.pickle.bz2'%(i,j), 'rb')
    #         data = pickle.load(file)
    #         file.close()
    #     # show_heat(data)
    #         show_lifespan(data)
    file = bz2.open('results/no_blinks/dataframes/gfeat/%s_eye%s.mp4_test.pickle.bz2'%(200,0), 'rb')
    data = pickle.load(file)
    file.close()
    print(data['avg_lifespan'])
    file = bz2.open('results/no_blinks/dataframes/gfeat/%s_eye%s.mp4_test.pickle.bz2'%(200,1), 'rb')
    data = pickle.load(file)
    file.close()
    print(data['avg_lifespan'])


# print(data.memory_usage())
# last = data.iloc[80]
# cv.imwrite('heat.png', last.loc['heatmap'])
# cv.waitKey(0)
# hist = last.loc['hist_lifespan']
# plt.hist(hist ,bins=int(max(hist) / 20))
# plt.show()

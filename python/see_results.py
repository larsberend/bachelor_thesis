#!/usr/bin/env python
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import sys
import bz2


for i in range(10):
    for k in range(2):
        file = bz2.open('results/no_blinks/dataframes/gfeat/%s_eye%s.mp4_test.pickle.bz2'%(i,k), 'rb')
        data = pickle.load(file)
        print(i)
        print(k)

        for ls, para in zip(data.head(5)['avg_lifespan'][:], data.head(5)['detector_params'][:]):
            print(ls)
            print(para)
        file.close()
# file = bz2.open('/home/laars/uni/BA/code/python/results/no_blinks/dataframes/testabc.pickle', 'wb')
# pickl.bz2e.dump(data['hist_lifespan'], file)
# file.close()
# sfile = bz2.BZ2File('/home/laars/uni/BA/code/python/results/no_blinks/dataframes/compress', 'w')
# pickle.dump(data, sfile)
# sfile.close()
file = bz2.open('results/no_blinks/dataframes/gfeat/%s_eye%s.mp4_test.pickle.bz2'%(200,0), 'rb')
data = pickle.load(file)
file.close()
for ls, para in zip(data.head(5)['avg_lifespan'][:], data.head(5)['detector_params'][:]):
    print(ls)
    print(para)
file = bz2.open('results/no_blinks/dataframes/gfeat/%s_eye%s.mp4_test.pickle.bz2'%(200,1), 'rb')
data = pickle.load(file)
print('hello')
file.close()
for ls, para in zip(data.head(5)['avg_lifespan'][:], data.head(5)['detector_params'][:]):
    print(ls)
    print(para)
# print(data.memory_usage())
# last = data.iloc[80]
# cv.imwrite('heat.png', last.loc['heatmap'])
# cv.waitKey(0)
# hist = last.loc['hist_lifespan']
# plt.hist(hist ,bins=int(max(hist) / 20))
# plt.show()


# print(type(data)==pd.core.frame.DataFrame)
# for ls, para in zip(data.head(10)['avg_lifespan'][:], data.head(10)['detector_params'][:]):
#     print(ls)
#     print(para)
# for heat in data.head(3)['heatmap']:
#     cv.imshow('heat.png', heat)
#     cv.waitKey(0)
# cv.imshow('last heat', last.loc['heatmap'])
# cv.waitKey(0)
# for hist in data.head(3)['hist_lifespan']:
#     print(hist.shape)
#     plt.hist(hist ,bins=int(max(hist) / 20))
#     plt.show()
# hist = last.loc['hist_lifespan']
# plt.hist(hist ,bins=int(max(hist) / 20))
# print(hist.shape)
# plt.show()
# sorted.iloc[6,:])
# for map in data.iloc[8,1:]:
#     if map is not None:
#         cv.imshow( 'as', map)
#         cv.waitKey(0)
# print(data.iloc[0]['hist_lifespan'].shape)
# lifespans = data.iloc[7,1:]

# hist_lifespan = np.asarray(hist_lifespan)
# print(hist_lifespan.shape())
# for hist_lifespan in lifespans:
#     if hist_lifespan is not None:
#         plt.hist(hist_lifespan ,bins=int(max(hist_lifespan) / 20))
#         plt.show()

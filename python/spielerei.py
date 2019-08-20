import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

cv.buildOpticalFlowPyramid()
im = cv.imread('/home/laars/uni/BA/code/python/results/none_frame/0.png',0)

kernel = cv.getGaborKernel((15,15), sigma = 1, theta = 1, lambd = 1, gamma = 1)

gabor = cv.filter2D(im, -1, kernel = kernel)
goodfeats = cv.goodFeaturesToTrack(im, maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)
print(np.unique(gabor, return_counts=True))
# print(gabor.shape)
# print(goodfeats)
# cv.imshow('Gabor', goodfeats)
# cv.destroyAllWindows()
cv.waitKey(0)

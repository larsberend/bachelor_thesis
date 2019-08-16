import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def main(path):
    img = cv.imread(path, cv.CV_8UC1)
    img = cv.equalizeHist(img)
    hist = cv.calcHist(img, [0], None,[256],[0,256])
    cv.imshow('img',img)
    plt.plot(hist)
    plt.show()
    cv.waitKey(0)



if __name__=='__main__':
    import sys
    try:
        path = sys.argv[1]
    except:
        print('Please specify an image!')
    main(path)

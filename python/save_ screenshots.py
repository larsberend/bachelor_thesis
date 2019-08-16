#!/usr/bin/env python

'''
Versucht mit 100 Frames, keine Verbesserung

'''

import deflicker
import cv2 as cv
import video
def main():
    cap = video.create_capture('/home/laars/uni/BA/eyes/Gold_Standard_Kopien/1/1d38e5a6-7b4b-49cd-b23c-8dc92b35c265/1d38e5a6-7b4b-49cd-b23c-8dc92b35c265/000/eye1.mp4')
    for x in range(0, 100):
        _ret, frame = cap.read()
        cv.imshow('frame',frame)
        key = cv.waitKey(0)
        if key == 13: # Enter
            cv.imwrite('/home/laars/uni/BA/git_pythonsample/python/results/wflicker/%s.png' % (x), frame)
        if key == 27: # Escape
            break
if __name__=='__main__':
    main()

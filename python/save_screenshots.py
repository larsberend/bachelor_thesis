#!/usr/bin/env python

'''
Save frames as images using Enter. Escape for exit.

'''

# import deflicker
import cv2 as cv
import video
def main():
    cap = video.create_capture('/home/laars/uni/BA/eyes/000/eye1.mp4')
    for x in range(0, 100):
        _ret, frame = cap.read()
        frame = cv.medianBlur(frame, 15)
        frame = cv.resize(frame, (480, 360))
        cv.imshow('frame',frame)
        key = cv.waitKey(0)
        if key == 13: # Enter
            cv.imwrite('/home/laars/uni/BA/code/python/results/wflicker/%s.png' % (x), frame)
        if key == 27: # Escape
            break
if __name__=='__main__':
    main()

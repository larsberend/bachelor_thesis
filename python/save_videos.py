import numpy as np
import cv2 as cv
import video
from midway_equal_imgs import midway_equal_imgs

# use midway equalization and save a new video.
# Finding a valid codec and file-extension for version of OS, OpenCV and Python is hard.
# see http://www.fourcc.org/codecs.php

nr_frames = 8
# path = '/home/laars/uni/BA/eyes/Gold_Standard_Kopien/1/1d38e5a6-7b4b-49cd-b23c-8dc92b35c265/1d38e5a6-7b4b-49cd-b23c-8dc92b35c265/000/eye1.mp4'
path = '/home/laars/uni/BA/eyes/200hzMessung/9c8d96f7-65e7-4172-baf6-73bbc63863af/9c8d96f7-65e7-4172-baf6-73bbc63863af/000/eye0.mp4'
cap = video.create_capture(path)
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)
# Define the codec and create VideoWriter object
# working: (MJPG, .avi), (mp4v, .mp4), ()

codec = 'mp4v'
fourcc = cv.VideoWriter_fourcc(*codec)
out = cv.VideoWriter('/home/laars/uni/BA/code/python/results/deflicker/eye0_midway_equal_200.mp4', fourcc, fps, (width, height), False)
end_frame = cap.get(7)
while cap.get(1) < end_frame-1:
    imgs = np.zeros((nr_frames, height, width))
    for x in range(nr_frames):
        ret, frame = cap.read()
        if cap.get(1)%1000 == 0:
            print(cap.get(1))
        if ret==True:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            imgs[x] = frame

    imgs_de = midway_equal_imgs(imgs, nr_frames=nr_frames, save_path=None)

    for deflick in imgs_de:
        out.write(deflick)

# Release everything if job is finished
print("while finised")
cap.release()
print("caprelease")

out.release()
print("out release")
cv.destroyAllWindows()

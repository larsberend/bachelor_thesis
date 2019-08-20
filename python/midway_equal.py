import numpy as np
import cv2 as cv
import video
import matplotlib.pyplot as plt

nr_frames = 3
path = '/home/laars/uni/BA/eyes/Gold_Standard_Kopien/1/1d38e5a6-7b4b-49cd-b23c-8dc92b35c265/1d38e5a6-7b4b-49cd-b23c-8dc92b35c265/000/eye1.mp4'#'/home/laars/uni/BA/eyes/000/eye1.mp4'


def midway_equal(   path = path,
                    nr_frames = nr_frames,
                    save_path='/home/laars/uni/BA/code/python/results/deflicker/'
                ):
    # frames = []
    indices = []
    cap = video.create_capture(path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    sum_sort = np.zeros((width * height))
    backsort = np.zeros((nr_frames, width* height))

    for x in range(nr_frames):
        _ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # cv.imshow('frame', frame)
        # cv.waitKey(0)

        # print(np.unique(frame, return_counts=True))
        # frames.append(frame)
        flat = np.concatenate(frame)
        indices.append(flat.argsort().argsort())
        print(len(indices[0]))
        # print(np.unique(np.sort(flat), return_counts=True))
        sort = np.sort(flat)
        # hist = cv.calcHist(frame, [0], None,[256],[0,256])
        # plt.plot(sort)
        # plt.show()
        sum_sort = sum_sort + sort
    mean = np.array(sum_sort/nr_frames, dtype=np.uint8)

    # print(np.unique(mean, return_counts=True))

    for y in range(nr_frames):
        print(y)
        backsort[y] = mean[indices[y]]
        frame_def = np.reshape(backsort[y], (height, width))
        # hist = cv.calcHist(frame_def, [0], None,[256],[0,256])
        # plt.plot(hist)
        # plt.show()


        # cv.imshow('frame_def', frame_def)
        # cv.waitKey(0)
        if save_path is not None:
            print('here')
            cv.imwrite('%s%s.png' % (save_path, y+1),
                        frame_def
                        )

if __name__== '__main__':
    midway_equal()

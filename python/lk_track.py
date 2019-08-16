#!/usr/bin/env python
# python3 lk_track.py /home/laars/uni/BA/eyes/000/eye1.mp4 0 1000
'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import video
from common import anorm2, draw_str
from time import clock

# params from example
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                  )
lk_pyr_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
                    )

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src, start_frame = 0, end_frame = None):
        self.track_len = 10
        self.detect_interval = 20
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.cam.set(1, start_frame)
        self.start_frame = start_frame

        # set start and end frame for terminal use
        self.frame_idx = int(start_frame)
        if end_frame is not None:
            self.end_frame = int(end_frame)
        else:
            self.end_frame = self.cam.get(7)
        self.length = self.end_frame - start_frame
    def run(self, interpolation = "yes", blur = "yes"):

        # counters for evvaluation
        none_idx = []
        feat_count = 0
        lifespan = []
        frame_size = (320, 240)
        heatmap = np.zeros(frame_size)


        while True:
            _ret, frame = self.cam.read()

            # von mir:
            # if interpolation is not None:
            #     frame = cv.resize(frame, frame_size, interpolation = cv.INTER_LINEAR)
                # print(frame.shape)
            # if blur is not None:
                # frame = cv.GaussianBlur(frame, (25,25), 0)
                # frame = cv.medianBlur(frame, 35)
                # print(frame.shape)



            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            # print(frame_gray.shape)
            # print(heatmap.shape)

            if len(self.tracks) > 0:
                # no tracks to follow: use forward and backward optical flow
                # to get start of a track.
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)


                # with pyramids:
                scales0, pyramid0 = cv.buildOpticalFlowPyramid(img0, (15, 15), 20)
                scales1, pyramid1 = cv.buildOpticalFlowPyramid(img1, (15, 15), 20)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(cv.UMat(pyramid0), cv.UMat(pyramid1),ranges = [1,2,3],  None, **lk_pyr_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(pyramid1, pyramid0, p1, None, **lk_pyr_params)

                # without pyramids:
                # p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)


                d = abs(p0-p0r)

                d = d.reshape(-1, 2)
                d = d.max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        lifespan.append(len(tr))
                        continue
                    tr.append((x, y))

                    heatmap[min(int(x), frame_size[0]-1), min(int(y), frame_size[1]-1)] += 1
                    # if len(tr) > self.track_len:
                    #     del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 3, (0, 255, 0), -1)

                self.tracks = new_tracks

                feat_count += len(self.tracks)

                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255, 255, 0))
                # draw_str(vis, (5, 5), 'track count: %d' % len(self.tracks))
            # if len(self.tracks) < 5:
            #     print('ok')
            #     none_idx.append(self.frame_idx)
            else:
                none_idx.append(self.frame_idx)

            if self.frame_idx % self.detect_interval == 0:

                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)

                p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                # hog_desc = cv.HOGDescriptor()
                # p = hog_desc.compute(frame_gray)


                if p is not None:
                    # print(len(p))
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                # else:
                #     print('none')
                    # none_idx.append(self.frame_idx)
                # else:
                #     print('none')
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)
            # print(vis.shape)
            ch = cv.waitKey(1)
            if ch == 27 or (self.cam.get(1) == self.end_frame):

                for tr in self.tracks:
                    lifespan.append(len(tr))

                print('All points tracked: %s. \nLength of video: %s frames. \nAverage points per frame: %s.'
                        % (len(lifespan),
                            self.frame_idx - self.start_frame,
                            sum(lifespan)/(self.frame_idx - self.start_frame)
                            )
                )
                print('\nLifespan Min: %s, Occurences: %s \nLifespanMax: %s, Occurences: %s\nAverage Lifespan: %s frames'
                        % (min(lifespan), lifespan.count(min(lifespan)),
                        max(lifespan), lifespan.count(max(lifespan)),
                        sum(lifespan)/len(lifespan)))
                print('feat_count: ' + str(feat_count))
                print('sum lifespan: ' + str(sum(lifespan)))
                heatmapshow = None
                heatmapshow = cv.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                heatmapshow = heatmapshow.transpose(1, 0)
                heatmapshow = cv.applyColorMap(heatmapshow, cv.COLORMAP_JET)

                # Auf Dezimalstellen runden, Speicherreduktion
                rounded = []
                for idx in none_idx:
                    rounded.append(round(idx, -1))

                for idx in set(rounded):
                    self.cam.set(1, idx)
                    _ret, none_frame = self.cam.read()
                    if not _ret:
                        continue
                    # cv.imwrite('/home/laars/uni/BA/git_pythonsample/python/results/none_frame/%s.png' % (idx), none_frame)


                cv.imshow('image', heatmapshow)
                cv.waitKey(0)
                cv.destroyAllWindows()
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    if len(sys.argv) == 3:
        app = App(video_src, int(sys.argv[2]))
    elif len(sys.argv) == 4:
        app = App(video_src, int(sys.argv[2]), int(sys.argv[3]))
    else:
        app = App(video_src)
    app.run(blur = True)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
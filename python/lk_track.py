#!/usr/bin/env python
# python3 lk_track.py /home/laars/uni/BA/eyes/000/eye1.mp4 0 1000
"""
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>] [/start_frame] [/end_frame]


Keys
----
ESC - exit

"""

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import video


detector = 'orb'       # {'good', 'orb', 'sift', 'surf'}


# params from example
lk_params = dict(winSize=(15, 15),
                 maxLevel=10,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                 )

gfeature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

orb_params = dict(  nfeatures=500,
                    scaleFactor=1.2,
                    nlevels=10,
                    edgeThreshold=100,
                    firstLevel=0,
                    WTA_K=2,
                    # scoreType='HARRIS_SCORE',
                    patchSize=31,
                    fastThreshold=20
                    )

def get_keypoints(frame, mask, detector, detector_params):
    return frame


def my_orb(img, mask):
    orb = cv.ORB_create(**orb_params)
    kps = orb.detect(img, mask=mask)
    return cv.KeyPoint_convert(kps)


def my_sift(img, mask):
    sift = cv.xfeatures2d.SIFT_create(nfeatures=0,
                                    nOctaveLayers=3,
                                    contrastThreshold=0.04,
                                    edgeThreshold=10,
                                    sigma=1.6
                                    )
    kps = sift.detect(img, mask=mask)
    return cv.KeyPoint_convert(kps)

def my_surf(img, mask):
    surf = cv.xfeatures2d.SURF_create(hessianThreshold = 100,
                                    nOctaves=4,
                                    nOctaveLayers=3,
                                    extended=False,
                                    upright=False
                                    )
    kps = surf.detect(img, mask=mask)
    return cv.KeyPoint_convert(kps)


class App:
    def __init__(self, video_src, start_frame=0, end_frame=None):
        self.track_len = 10
        self.detect_interval = 20
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.cam.set(1, start_frame)
        self.start_frame = start_frame
        self.prev_gray = None
        # set start and end frame for terminal use
        self.frame_idx = int(start_frame)
        if end_frame is not None:
            self.end_frame = int(end_frame)
        else:
            self.end_frame = self.cam.get(7)
        self.length = self.end_frame - start_frame

    def run(self, detector='good'):

        # counters for evaluation
        none_idx = []
        feat_count = 0
        lifespan = []
        frame_size = (int(self.cam.get(3)), int(self.cam.get(4)))
        heatmap = np.zeros(frame_size)

        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray

                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0 - p0r)
                d = d.reshape(-1, 2).max(-1)
                good = d < 1

                new_tracks = []

                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        lifespan.append(len(tr))
                        continue
                    tr.append((x, y))
                    heatmap[min(int(x), frame_size[0] - 1), min(int(y), frame_size[1] - 1)] += 1

                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 3, (0, 255, 0), -1)

                self.tracks = new_tracks
                feat_count += len(self.tracks)
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255, 255, 0))
            else:
                none_idx.append(self.frame_idx)

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                cv.imshow('mask', mask)
                cv.waitKey(0)

                if detector == 'good':
                    p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **gfeature_params)
                elif detector == 'orb':
                    p = my_orb(frame_gray, mask)
                elif detector == 'sift':
                    p = my_sift(frame_gray, mask)
                elif detector == 'surf':
                    p = my_surf(frame_gray, mask)

                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            cv.imshow('lk_track', vis)
            ch = cv.waitKey(1)

            if ch == 27 or (self.cam.get(1) == self.end_frame):
                for tr in self.tracks:
                    lifespan.append(len(tr))

                print('All points tracked: %s. \nLength of video: %s frames. \nAverage points per frame: %s.'
                      % (len(lifespan),
                         self.frame_idx - self.start_frame,
                         sum(lifespan) / (self.frame_idx - self.start_frame)
                         )
                      )
                if len(lifespan) > 0:
                    print(
                        '\nLifespan Min: %s, Occurences: %s \nLifespanMax: %s, Occurences: %s\nAverage Lifespan: %s frames'
                        % (min(lifespan), lifespan.count(min(lifespan)),
                           max(lifespan), lifespan.count(max(lifespan)),
                           sum(lifespan) / len(lifespan)))
                    print('feat_count: ' + str(feat_count))
                    print('sum lifespan: ' + str(sum(lifespan)))

                heatmapshow = cv.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                           dtype=cv.CV_8U)
                heatmapshow = heatmapshow.transpose(1, 0)
                heatmapshow = cv.applyColorMap(heatmapshow, cv.COLORMAP_HOT)

                lifespan_hist = np.asarray(np.unique(lifespan, return_counts=True)).transpose(1, 0)
                plt.hist(lifespan_hist, bins=np.arange(lifespan_hist.min(), lifespan_hist.max() + 1))
                plt.show()

                # Auf Dezimalstellen runden, Speicherreduktion
                rounded = []
                for idx in none_idx:
                    rounded.append(round(idx, -1))

                for idx in set(rounded):
                    self.cam.set(1, idx)
                    _ret, none_frame = self.cam.read()
                    if not _ret:
                        continue
                    # cv.imwrite('/home/laars/uni/BA/code/python/results/none_frame/%s.png' % (idx), none_frame)

                cv.imwrite('heatmap.png', heatmapshow)
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
    app.run(detector)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

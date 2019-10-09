#!/usr/bin/env python
# python3 tracking.py /home/laars/uni/BA/eyes/000/eye1.mp4 0 1000
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
from math import degrees, atan2
import pandas as pd

import hog
import video

# params from example
lk_params = dict(winSize=(15, 15),
                 maxLevel=10,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                 )
gfeature_params = dict(maxCorners=200,
                       qualityLevel=0.6,
                       minDistance=7,
                       blockSize=7
                       )
orb_params = dict(nfeatures=500,
                  scaleFactor=1.2,
                  nlevels=10,
                  edgeThreshold=100,
                  firstLevel=0,
                  WTA_K=2,
                  scoreType=0, # 0= HARRIS_SCORE, 1= FAST_SCORE
                  patchSize=31,
                  fastThreshold=20
                  )
sift_params = dict(nfeatures=500,
                   nOctaveLayers=3,
                   contrastThreshold=0.04,
                   edgeThreshold=10,
                   sigma=1.6
                   )
surf_params = dict(hessianThreshold=100,
                   nOctaves=4,
                   nOctaveLayers=3,
                   extended=False,
                   upright=False
                   )

detector = 'good'  # {'good', 'orb', 'sift', 'surf'}
detector_params = gfeature_params


def get_Angle(pointA, pointB):
    # angle of line1: (Ax-1, Ay) Point A, and
    #          line2: (Point A, Point B), counterclockwise
    # Go from point A in the following directions,to find point B.
    # If the angle is:
    # 0° : "down"
    # 90° : "right"
    # 180° : "up"
    # 270° : "left"
    y = pointB[1] - pointA[1]
    x = pointB[0] - pointA[0]
    return degrees(atan2(y, x)) % 360

def get_keypoints(frame, mask, detector, detector_params):
    if detector == 'good':
        kps = cv.goodFeaturesToTrack(frame, mask=mask, **gfeature_params)
        return kps
    elif detector == 'orb':
        detec = cv.ORB_create(**detector_params)
    elif detector == 'sift':
        detec = cv.xfeatures2d.SIFT_create(**detector_params)
    elif detector == 'surf':
        detec = cv.xfeatures2d.SURF_create(**detector_params)
    kps = detec.detect(frame, mask=mask)
    return cv.KeyPoint_convert(kps)


class keypoint_tracker:
    def __init__(self, video_src, start_frame=0, end_frame=None):
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.cam.set(1, start_frame)
        self.start_frame = start_frame
        self.prev_gray = None
        # set start and end frame for  use
        self.frame_idx = int(start_frame)
        self.end_frame = self.cam.get(7) if end_frame is None else end_frame
        self.length = self.end_frame - start_frame
    def run(self,
            detector='good',
            detector_params=gfeature_params,
            lk_params=lk_params,
            resize=(3/4, 3/4),
            filterType='median',
            filterSize=(15,15),
            printing=True,
            angMag = False):

        # counters for evaluation
        none_idx = []
        # all_tracks = pd.DataFrame(index=['frameNr_' + str(i) for i in range(self.start_frame, self.end_frame)])
        all_tracks = {}
        track_num = 0
        frame_size = (int(self.cam.get(3)*resize[0]), int(self.cam.get(4)*resize[1]))  # (width, height)
        heatmap = np.zeros(frame_size)

        while True:
            if self.frame_idx%1000 == 0:
                print(self.frame_idx)
            _ret, frame = self.cam.read()
            if filterType == 'median':
                frame = cv.medianBlur(frame, filterSize[0])
            frame = cv.resize(frame, frame_size)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if not self.tracks:
                none_idx.append(self.frame_idx)

            else:
                img0, img1 = self.prev_gray, frame_gray

                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0 - p0r)
                d = d.reshape(-1, 2).max(-1)
                good = d < 1
                # good = np.logical_not(good)
                # good = st
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        if len(tr) > 1:
                            # all_tracks[track_num] = None
                            # all_tracks.iloc[self.frame_idx - len(tr): self.frame_idx, track_num] = tr
                            all_tracks.setdefault(self.frame_idx-len(tr),[]).append(tr)
                        track_num += 1
                        continue
                    tr.append((x, y))
                    heatmap[min(int(x), frame_size[0] - 1),
                            min(int(y), frame_size[1] - 1)] += 1

                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 3, (0, 255, 0), -1)

                self.tracks = new_tracks
                if printing:
                    cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255, 255, 0))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.full(frame_gray.shape, fill_value=255, dtype=np.uint8)
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)

                p = get_keypoints(frame_gray, mask, detector, detector_params)

                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            if printing:
                cv.imshow(detector, vis)

            ch = cv.waitKey(1)
            if ch == 27 or (self.cam.get(1) == self.end_frame):
                # print('End of video reached.')
                for tr in self.tracks:
                    # all_tracks[track_num] = np.nan
                    # all_tracks.iloc[self.frame_idx - len(tr): self.frame_idx, track_num] = tr
                    all_tracks.setdefault(self.frame_idx-len(tr),[]).append(tr)
                    track_num += 1

                avg_lifespan, hist_lifespan, heatmap = self.get_results(all_tracks,track_num, heatmap, show_me=printing)
                print(avg_lifespan)
                hod = None
                if angMag:
                    hod, angles, magnitudes = self.AngMag(tracks)

                cv.destroyAllWindows()
                self.cam.release()
                if hod is None:
                    return [avg_lifespan, hist_lifespan, heatmap, none_idx]
                else:
                    return [avg_lifespan, hist_lifespan, heatmap, hod, none_idx]

    def AngMag(self, tracks, return_df=True, ho_bins=8):
        angles = np.full((tracks.shape[0], tracks.shape[1]), np.nan)
        mags = np.copy(angles)

        hogs = np.full((tracks.shape[0], ho_bins), np.nan)
        for nr_row in range(tracks.shape[0] - 1):
            for nr_col in range(tracks.shape[1]):
                a = tracks[nr_row, nr_col]
                b = tracks[nr_row + 1][nr_col]
                if a is not None and b is not None:
                    angle = get_Angle(a, b)
                    angles[nr_row + 1, nr_col] = angle
                    mag = np.linalg.norm(np.array(a) - np.array(b))
                    mags[nr_row + 1, nr_col] = mag

            ang_li = [x for x in list(angles[nr_row + 1]) if not np.isnan(x)]
            mag_li = [x for x in list(mags[nr_row + 1]) if not np.isnan(x)]
            hogs[nr_row + 1, :] = hog.hog(ang_li, mag_li, ho_bins)
        hogs = hogs[1:]
        # print(np.unique(hogs[1:], return_counts=True))
        hog_norm = cv.normalize(hogs, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                    dtype=cv.CV_8U).transpose(1, 0)
        hogshow = cv.applyColorMap(hog_norm, cv.COLORMAP_SUMMER)

        #cv.imwrite('directions.png',hogshow)
            #
            # print(ang_li,"\nmag\n")
            # print(mag_li,"\nhogs\n")
            # print(hogs[nr_row+1])

        # print("tracks\n", tracks,"\hogs\n", np.unique(hogs[1:,:], return_counts=True),'\nang\n',angles.shape,"\nmag\n", mags.shape)
        return hogshow

    def get_results(self, tracks, track_num, heatmap, show_me=False, none_idx=None):
        # print('Calculating results...')
        if not tracks:
            print('No points found or tracked!')
        else:
            np_tracks = np.zeros((track_num, self.end_frame, 2), dtype=np.float32)
            tr_num = 0
            lifespan = []
            for key in tracks:
                for x in range(len(tracks[key])):
                    lifespan.append(len(tracks[key][x]))
                    np_tracks[tr_num, key : key + len(tracks[key][x])]=tracks[key][x]
                    tr_num += 1
            tracks = np_tracks

            lifespan_hist = np.round(np.array(lifespan), -1)
            heatmap_norm = cv.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                        dtype=cv.CV_8U).transpose(1, 0)
            heatmapshow = cv.applyColorMap(heatmap_norm, cv.COLORMAP_SUMMER)

            if show_me:
                print('All points tracked: %s. \nLength of video: %s frames. \nAverage points per frame: %s.'
                      % (len(lifespan),
                         self.frame_idx - self.start_frame,
                         sum(lifespan) / (self.frame_idx - self.start_frame)
                         )
                      )
                print(
                    '\nLifespan Min: %s, Occurences: %s \nLifespanMax: %s, Occurences: %s\nAverage Lifespan: %s frames'
                    % (min(lifespan), lifespan.count(min(lifespan)),
                       max(lifespan), lifespan.count(max(lifespan)),
                       sum(lifespan) / len(lifespan)))

                print('sum lifespan: ' + str(sum(lifespan)))

                cv.imwrite('heatmap.png', heatmapshow)
                plt.hist(lifespan_hist, bins=int(max(lifespan_hist) / 20))
                plt.show()

            # Frames ohne Keypoints (=blinks?)
            # auf Dezimalstellen runden, Speicherreduktion
            if none_idx is not None:
                idx_round = [round(idx, -1) for idx in none_idx]
                for idx in set(idx_round):
                    self.cam.set(1, idx)
                    _ret, none_frame = self.cam.read()
                    if not _ret:
                        continue
                    cv.imwrite('/home/laars/uni/BA/code/python/results/none_frame/%s.png' % (idx), none_frame)

            return sum(lifespan) / len(lifespan), lifespan_hist, heatmapshow


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    if len(sys.argv) == 3:
        kt = keypoint_tracker(video_src, int(sys.argv[2]))
    elif len(sys.argv) == 4:
        kt = keypoint_tracker(video_src, int(sys.argv[2]), int(sys.argv[3]))
    else:
        kt = keypoint_tracker(video_src)
    kt.run(detector, detector_params)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

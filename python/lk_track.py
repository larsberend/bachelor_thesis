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
from math import atan, atan2, degrees
import pandas as pd

import hog
import video

# params from example
lk_params = dict(winSize=(15, 15),
                 maxLevel=10,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                 )

gfeature_params = dict(maxCorners=500,
                       qualityLevel=0.3,
                       minDistance=7,
                       blockSize=7
                      )
orb_params = dict(nfeatures=500,
                  scaleFactor=1.2,
                  nlevels=10,
                  edgeThreshold=100,
                  firstLevel=0,
                  WTA_K=2,
                  # scoreType='HARRIS_SCORE',
                  patchSize=31,
                  fastThreshold=20
                  )
sift_params = dict(nfeatures=0,
                   nOctaveLayers=3,
                   contrastThreshold=0.04,
                   edgeThreshold=10,
                   sigma=1.6
                   )
surf_params = dict(hessianThreshold = 100,
                   nOctaves=4,
                   nOctaveLayers=3,
                   extended=False,
                   upright=False
                   )

detector = 'sift'       # {'good', 'orb', 'sift', 'surf'}
detector_params = sift_params


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


class App:
    def __init__(self, video_src, start_frame=0, end_frame=None):
        self.track_len = 10
        self.detect_interval = 2
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.cam.set(1, start_frame)
        self.start_frame = start_frame
        self.prev_gray = None
        # set start and end frame for  use
        self.frame_idx = int(start_frame)
        if end_frame is not None:
            self.end_frame = int(end_frame)
        else:
            self.end_frame = self.cam.get(7)
        self.length = self.end_frame - start_frame

    def run(self,
            detector='good',
            detector_params=gfeature_params,
            printing=False):

        # counters for evaluation
        none_idx = []
        # all_tracks = {}
        all_tracks = pd.DataFrame(index=['frameNr_' + str(i) for i in range(self.start_frame, self.end_frame)])
        track_num = 0
        frame_size = (int(self.cam.get(3)), int(self.cam.get(4))) # (width, height)
        heatmap = np.zeros(frame_size)

        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_gray = cv.medianBlur(frame_gray, 25)
            vis = frame.copy()
            if self.tracks:
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
                        try:
                            all_tracks[track_num]
                        except KeyError:
                            all_tracks[track_num]=None
                        all_tracks.loc['frameNr_' + str(self.frame_idx-len(tr)) : 'frameNr_' + str(self.frame_idx-1), track_num] = tr
                        track_num += 1
                        continue
                    tr.append((x, y))
                    heatmap[min(int(x), frame_size[0] - 1),
                            min(int(y), frame_size[1] - 1)] += 1

                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 3, (0, 255, 0), -1)

                self.tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255, 255, 0))
            else:
                none_idx.append(self.frame_idx)

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = get_keypoints(frame_gray, mask, detector, detector_params)

                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            cv.imshow(detector, vis)
            ch = cv.waitKey(1)

            if ch == 27 or (self.cam.get(1) == self.end_frame):

                for tr in self.tracks:
                    try:
                        all_tracks[track_num]
                    except KeyError:
                        all_tracks[track_num] = None
                    all_tracks.loc['frameNr_' + str(self.frame_idx-len(tr)) : 'frameNr_' + str(self.frame_idx-1), track_num] = tr
                    track_num += 1
                heatmap, lifespan_hist, avg_lifespan = self.get_results(all_tracks, heatmap, printing)
                cv.destroyAllWindows()
                return avg_lifespan, detector, detector_params, all_tracks, heatmap, lifespan_hist

    def AngMag(self, tracks, return_df = True, ho_bins=8):
        angles = pd.DataFrame().reindex_like(tracks)
        mags = pd.DataFrame().reindex_like(tracks)

        hogs = pd.DataFrame(index = tracks.index, columns=np.arange(0, ho_bins))
        for nr_row in range(tracks.shape[0]-1):
            for nr_col in range(tracks.shape[1]):
                a = tracks.iloc[nr_row, nr_col]
                b = tracks.iloc[nr_row+1][nr_col]
                if a is not None and b is not None:
                    angle = get_Angle(a,b)
                    angles.iloc[nr_row+1, nr_col] = angle
                    mag = np.linalg.norm(np.array(a)-np.array(b))
                    mags.iloc[nr_row+1, nr_col] = mag
            ang_li = [x for x in list(angles.iloc[nr_row+1]) if not np.isnan(x)]
            mag_li = [x for x in list(mags.iloc[nr_row+1]) if not np.isnan(x)]
            hogs.iloc[nr_row+1] = hog.hog(ang_li, mag_li, ho_bins)



        return hogs, angles, mags

    def get_results(self, tracks, heatmap, show_me=False, none_idx=None):
        if tracks.empty:
            print('No points found or tracked!')
            return None, None, None
        else:
            lifespan = []
            # get angles and magnitudes of tracks
            hogs, angles, magnitudes = self.AngMag(tracks)
            print(hogs,'\n', angles,'\n', magnitudes)
            # li = list(angles.loc['frameNr_5'])
            # li = [x for x in li if not np.isnan(x)]
            #
            # li2 = list(magnitudes.loc['frameNr_5'])
            # li2 = [y for y in li2 if not np.isnan(y)]
            #
            # ho = hog.hog(li,li2, 8)
            # print(type(ho))
        # for row in tracks.itertuples(index=False):



            # for start_fr, tr_nr in tracks.iteritems():
            #     print(start_fr)
            #     print('\ntr_nr')
            #     print(tr_nr[start_fr])
            #     if tr_nr[start_fr]:
            #         for l in range(len(tr_nr[start_fr])-1):
            #             # angles[start_fr][tr_nr]=
            #             print(get_Angle(tr_nr[start_fr][l], tr_nr[start_fr][l+1]))
            # angles = {}.fromkeys(tracks.keys(),[])
            # magnitudes = {}.fromkeys(tracks.keys(),[])
            # for key, value in tracks.items():
            #     for li in value:
            #         track_angle = []
            #         track_mag = []
            #         for x in range(len(li)-1):
            #             lifespan.append(len(li))
            #             track_angle.append(get_Angle(li[x], li[x+1]))
            #             track_mag.append(np.linalg.norm(np.array(li[x])-np.array(li[x+1])))
            #
            #         angles[key].append(track_angle)
            #         magnitudes[key].append(track_mag)
                # for track in keys:
                #     track_angle = []
                #     track_mag = []
                #     for x in range(len(track)-1):
                #         # print(track[x]-track[x+1])
                #         track_mag.append(np.linalg.norm(np.array(track[x])-np.array(track[x+1])))
                #         track_angle.append(get_Angle(track[x], track[x+1]))
                #     angles.append(track_angle)
                #     magnitudes.append(track_mag)
                # hist_mag_ang = hog.hog(angles, magnitudes, self.frame_idx - self.start_frame)
            # print('tracks: \n')
            # for y in range(10):
            #     print(tracks[max_index][y])
            # print('\nmagnitudes \n')
            # print([magnitudes[max_index][z] for z in range(10)])
            #
            # print('\nangles\n')
            # print([angles[max_index][i]for i in range(10)])


            rounded_lifespan = []
            for length in lifespan:
                rounded_lifespan.append(round(length, -1))
            lifespan_hist = rounded_lifespan

            heatmap_norm = cv.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                       dtype=cv.CV_8U).transpose(1, 0)
            heatmapshow = cv.applyColorMap(heatmap_norm, cv.COLORMAP_HOT)

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


                cv.imshow('heatmap', heatmapshow)
                cv.waitKey(0)


                plt.hist(lifespan_hist, bins=int(max(lifespan_hist)/10))
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
            return heatmapshow, lifespan_hist, sum(lifespan) / len(lifespan)

def get_Angle(pointA, pointB):
    # angle of line1: (Ax-1, Ay) Point A, and
    #          line2: (Point A, Point B), counterclockwise
    # Go from point A in the following directions,to find point B.
    # If the angle is:
    # 0° : "down"
    # 90° : "right"
    # 180° : "up"
    # 270° : "left"
    y = pointB[1]-pointA[1]
    x = pointB[0]-pointA[0]
    return degrees(atan2(y, x)) % 360

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
    app.run(detector, detector_params)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

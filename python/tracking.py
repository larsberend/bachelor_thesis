#!/usr/bin/env python
"""
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow. Key point detection via Shi-Tomasi,
fast Hessian and DoG is implemented with
for track initialization and back-tracking for match verification
between frames.

Usage
-----
tracking.py [<video_source>] [/start_frame] [/end_frame]


Keys
----
ESC - exit

"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import degrees, atan2
import pandas as pd

import hog
import video

# For easy usage, example parameters are specified here. 
# For different ones, change them here or call function with different ones.


lk_params = dict(winSize=(31, 31),
                 maxLevel=3,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 4, 0.5),
                 )
gfeature_params = dict(maxCorners=510,
                       qualityLevel=0.5,
                       minDistance=61,
                       blockSize=5
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

    # print('pointA: %s'%(pointA.dtype))
    # print('pointB: %s'%(pointB.dtype))
    pointA = np.float32(pointA)
    pointB = np.float32(pointB)

    y = pointB[1] - pointA[1]
    x = pointB[0] - pointA[0]
    # print('y: %s'%(y))
    # print('x: %s'%(x))
    return degrees(atan2(y, x)) % 360


def get_keypoints(frame, mask, detector, detector_params):
    if detector == 'good':
        kps = cv.goodFeaturesToTrack(frame, mask=mask, **detector_params)
        return kps
    elif detector == 'sift':
        detec = cv.xfeatures2d.SIFT_create(**detector_params)
    elif detector == 'surf':
        detec = cv.xfeatures2d.SURF_create(**detector_params)
    kps = detec.detect(frame, mask=mask)
    return cv.KeyPoint_convert(kps)


class keypoint_tracker:
    def __init__(self, video_src, start_frame=None, end_frame=None):
        self.detect_interval = 10
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0 if start_frame is None else int(start_frame)
        self.start_frame = self.frame_idx
        self.cam.set(1, self.start_frame)
        self.prev_gray = None
        # set start and end frame for  use
        self.end_frame = int(self.cam.get(7)) if end_frame is None else end_frame
        self.length = self.end_frame - self.start_frame


    # use this to get measures from eye-videos like average lifespan, frames with no key points, 
    # heatmaps, histograms and seeing the video with the tracked points (printing=True). 
    # Also get heatmap histogram of directions (angMag=True).
    def run(self,
            detector=detector,
            detector_params=detector_params,
            lk_params=lk_params,
            resize=(3/4, 3/4),
            filterType='median',
            filterSize=(15,15),
            printing=True,
            angMag = False):
        print(detector_params)
        # counters for evaluation
        none_idx = []
        # all_tracks = pd.DataFrame(index=['frameNr_' + str(i) for i in range(self.start_frame, self.end_frame)])
        all_tracks = {}
        lifespans = []
        track_num = 0
        frame_size = (int(self.cam.get(3)*resize[0]), int(self.cam.get(4)*resize[1]))  # (width, height)
        heatmap = np.zeros(frame_size)
	
	# main loop, ends if end of video is reached or 'esc' is pressed, if a video is on the screen.
        while True:
            if self.frame_idx%1000 == 0:
                print(self.frame_idx)
            _ret, frame = self.cam.read()
            if filterType == 'median':
                frame = cv.medianBlur(frame, filterSize[0])
            elif filterType == 'gauss':
                frame = cv.GaussianBlur(frame, filterSize, sigmaX=0)
            frame = cv.resize(frame, frame_size)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()


            if not self.tracks:
                none_idx.append(self.frame_idx)

            else:
                img0, img1 = self.prev_gray, frame_gray
                # calculate LK- optical flow forwards and backwards, and take locations, which have been found 
		# in both directions.

                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0 - p0r)
                d = d.reshape(-1, 2).max(-1)
                good = d < 2
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        if len(tr) > 1:
                            if angMag:
                                all_tracks.setdefault(self.frame_idx-len(tr),[]).append(tr)
                            else:
                                lifespans.append(len(tr))
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
                    if angMag:
                        all_tracks.setdefault(self.frame_idx-len(tr),[]).append(tr)
                    else:
                        lifespans.append(len(tr))
                    track_num += 1

                if angMag:
                    lifespans, tracks = self.calc_lifespans(all_tracks, track_num)
                    hod = self.AngMag(tracks)
                if lifespans:
                    avg_lifespan, hist_lifespan, heatmap = self.get_results(lifespans, heatmap, show_me=printing)
                else:
                    avg_lifespan, hist_lifespan, heatmap = None, None, None
                print(len(lifespans))
                cv.destroyAllWindows()
                self.cam.release()
                if angMag:
                    return [avg_lifespan, hist_lifespan, heatmap, hod, none_idx]
                else:
                    return [avg_lifespan, hist_lifespan, heatmap, none_idx]


    def calc_lifespans(self, tracks, track_num):
        if not tracks:
            print('No points found or tracked!')
            return [None]
        else:
            np_tracks = np.zeros((track_num, self.end_frame, 2), dtype=np.uint16)
            tr_num = 0
            lifespan = []
            for key in tracks:
                for x in range(len(tracks[key])):
                    lifespan.append(len(tracks[key][x]))
                    np_tracks[tr_num, key : key + len(tracks[key][x])] = np.uint16(tracks[key][x])
                    tr_num += 1

            # print(np_tracks)
            return lifespan, np_tracks

    # return the results and print some of them, if specified.
                
    def get_results(self, lifespan, heatmap, show_me=False, none_idx=None):
            lifespan_hist = np.round(np.array(lifespan), -1)
            heatmap_norm = cv.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                        dtype=cv.CV_8U).transpose(1, 0)
            heatmapshow = cv.applyColorMap(heatmap_norm, cv.COLORMAP_COOL)

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
    
    # calculate HoD and save as .png
    def AngMag(self, tracks, return_df=True, ho_bins=8):
        # tracks = np.array([[[100,0],[0,0],[10,10],[100,100]],[[0,0],[30,50],[5,10],[10,100]]])
        # tracks = np.array([[[0,100],[0,0],[0,0],[0,0]],[[30,200],[30,50],[0,0],[0,0]]])
        tracks = np.array([[[50, 50],[60,50],[65,55],[65,65],[60,70],[50,70],[45,65],[45, 55], [50,50]]])
        angles = np.full((tracks.shape[0], tracks.shape[1]), 0)
        mags = np.copy(angles)
        # print(tracks)
        # print(tracks.shape)
        hogs = np.full((tracks.shape[1], ho_bins), np.nan)
        # print(hogs.dtype)
        # nr_row = nr of track
        # nr_col = pos of track
        for nr_row in range(tracks.shape[1]-1):
            for nr_col in range(tracks.shape[0]):
                a = tracks[nr_col, nr_row]
                b = tracks[nr_col][nr_row+1]
                if a.any() != 0 and b.any() != 0:
                    angle = get_Angle(a, b)
                    angles[nr_col, nr_row+1] = angle
                    mag = np.linalg.norm(np.array(a) - np.array(b))
                    mags[nr_col, nr_row+1] = mag
            hogs[nr_row] = hog.hog(angles[:, nr_row], mags[:, nr_row], ho_bins)

        hogs = hogs[1:]
        print('hogsende: ')
        hog_norm = cv.normalize(hogs, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                    dtype=cv.CV_8U).transpose(1, 0)
        hogshow = cv.applyColorMap(hog_norm, cv.COLORMAP_SUMMER)

        cv.imwrite('directions.png',hogshow)
        return hogshow

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


def blink_detection():
    video_path = '/home/laars/uni/BA/eyes/BlinkDataset/6_eye0.mp4'

    results = [video_path, [detector]]
    kt = keypoint_tracker(video_path, 0, 10)
    results.extend(kt.run(angMag = True))


    cv.destroyAllWindows()

def flicker_test():
    video_path = '/home/laars/uni/BA/code/python/results/deflicker/eye_1_midway_equal.mp4'
    kt = keypoint_tracker(video_path, 0, 10000)
    kt.run(printing=True, angMag = False)

def flicker_test200():
    video_path = '/home/laars/uni/BA/code/python/results/deflicker/eye0_midway_equal_200_8frames.mp4'
    lk = dict(winSize=(31, 31),
                     maxLevel=3,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                     )
    resize = (1,1)
    filterT = 'None'
    filterS = 'None'
    surf_params = dict(hessianThreshold=100,
                       nOctaves=1,
                       nOctaveLayers=5
                       )
    kt = keypoint_tracker(video_path, 0, 10000)
    kt.run(detector = 'surf',
           detector_params=surf_params,
           lk_params=lk,
           resize=resize,
           filterType=filterT,
           filterSize=filterS,
           printing=True,
           angMag=False
           )



if __name__ == '__main__':
    print(__doc__)
    main()
    # blink_detection()
    # flicker_test200()

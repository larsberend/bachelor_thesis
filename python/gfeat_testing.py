#!/usr/bin/env python
'''
Script to test various feature trackers located in tracking.py
The following data is saved as a Dataframe in the specified location.
1. Average Lifespan of tracks
2. Histogram of lifespans of tracks
3. Heatmap of location of tracks
4. Histogram of directions of movement of tracks
'''
import cv2 as cv
import numpy as np
import pandas as pd

import tracking as tr

video_path = '/home/laars/uni/BA/eyes/000/eye1.mp4'
save_path = 'results/dataframes/'

start = 450
end = 550
print('here')
all_data = {'detector_params':[], 'video_path':[],'start_frame':[], 'end_frame':[], 'avg_lifespan':[], 'hist_lifespan':[], 'heatmap':[], 'hod':[]}
for lvl in np.arange(0, 1, 0.2):
    print(lvl)
    for dist in np.arange(0, 100, 20):
        print(dist)
        for siz in np.arange(0, 10 , 2):
            for use in np.arange(0,2,1):
                gfeature_params = dict(maxCorners=500,
                                       qualityLevel=lvl,
                                       minDistance=dist,
                                       blockSize=siz,
                                       useHarrisDetector=use
                                       )

                kt = tr.keypoint_tracker(video_path, start_frame= start, end_frame=end)
                detector, detector_params, avg_lifespan, hist_lifespan, heatmap, hod = kt.run(detector='good', detector_params=gfeature_params)
                data = {
                        'detector_params' : [gfeature_params],
                        'video_path' : [video_path],
                        'start_frame': [start],
                        'end_frame': [end],
                        'avg_lifespan' : [avg_lifespan],
                        'hist_lifespan' : [hist_lifespan],
                        'heatmap' : [heatmap],
                        'hod' : [hod]
                        }

                for ind in data.keys():

                    all_data[ind].append(data[ind])
print('here 2')
res_df = pd.DataFrame(all_data) #,index=['detector_params', 'video_path', 'end_frame', 'avg_lifespan', 'hist_lifespan', 'heatmap', 'hod'])
res_df.to_csv(save_path + detector + str(start) + ' '+ str(end) + '.csv')
print('done')
print(res_df)

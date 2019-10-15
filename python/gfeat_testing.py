#!/usr/bin/env python
'''
Script to test various feature trackers located in tracking.py
The following data is saved as a Dataframe in the specified location:

video_path,
detector_params,
lk_params,
filterType,
filterSize,
resize,
avg_lifespan,
hist_lifespan,
heatmap,
none_idx,
runtime

Choose from following files:
0_eye0.mp4
0_eye1.mp4
1_eye0.mp4
1_eye1.mp4
2_eye0.mp4
2_eye1.mp4
3_eye0.mp4
3_eye1.mp4
4_eye0.mp4
4_eye1.mp4
5_eye0.mp4
5_eye1.mp4
6_eye0.mp4
6_eye1.mp4
7_eye0.mp4
7_eye1.mp4
8_eye0.mp4
8_eye1.mp4
9_eye0.mp4
9_eye1.mp4
200_eye0.mp4
200_eye1.mp4

'''
import cv2 as cv
import numpy as np
import pandas as pd
import sklearn.model_selection as selection
import timeit
import os
import sys
import pickle

import tracking as tr

print(__doc__)
try:
    video = sys.argv[1]
except:
    raise AssertionError('No video specified!')

folder = '/home/laars/uni/BA/eyes/Gold_Standard_Kopien/videos/'
save_path = '/home/laars/uni/BA/code/python/results/no_blinks/dataframes/'

start = 0
end = None
lk_params = dict(winSize=(15, 15),
                 maxLevel=10,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                 # termination criteria, epsilon AND count of iterations is used.
                 )

filterType='median'
filterSize=(21,21)
resize = (3/4, 3/4)

grid = selection.ParameterGrid(dict(maxCorners=np.arange(10, 511, 250),
                                    qualityLevel=np.array([0.01, 0.5, 0.99]),
                                    minDistance=np.arange(1, 62, 30),
                                    blockSize=np.arange(5, 36, 10)
                                    )
                                )
all_data = pd.DataFrame(index = range(len(grid)),
                      columns = ['video_path',
                                 'detector_params',
                                 'lk_params',
                                 'filterType',
                                 'filterSize',
                                 'resize',
                                 'avg_lifespan',
                                 'hist_lifespan',
                                 'heatmap',
                                 'none_idx',
                                 'runtime'
                                 ]
                        )

video_path = folder + video
print('video_path: ' + video_path)
for y in range(len(grid)):
    print('Job %s of %s' %(str(y), str(len(grid))))
    kt = tr.keypoint_tracker(video_path, start_frame=start, end_frame=end)
    results = [video_path, [grid[y]], [lk_params], filterType, filterSize, resize]
    start_t = timeit.default_timer()
    # detector, detector_params, avg_lifespan, hist_lifespan, heatmap = kt.run(detector='good', detector_params=params)
    results.extend(kt.run(  detector='good',
                            detector_params=grid[y],
                            lk_params=lk_params,
                            resize=resize,
                            filterType=filterType,
                            filterSize=filterSize,
                            printing=False,
                            angMag = False
                            )
                    )

    end_t = timeit.default_timer()
    results.append(end_t-start_t)
    print('runtime: ')
    print(end_t-start_t)
    for x in range(len(results)):
        all_data.iloc[y,x] = results[x]
print('sorting by avg_lifespan...')
sorted_data = all_data.sort_values(by='avg_lifespan', ascending=False)
print('saving to: '+ save_path + video  + '_test' + '.pickle...')
file = open(save_path + video  + '_test' + '.pickle', 'wb')
pickle.dump(sorted_data, file)
file.close()
print('done')

#!/usr/bin/env python
'''
Script to test fast Hessian features located in tracking.py
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


For more info see lk_testing.py

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
save_path = '/home/laars/uni/BA/code/python/results/no_blinks/dataframes/fhess/'

start = 4590
end = 4600
lk_params = dict(winSize=(15, 15),
                 maxLevel=10,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                 # termination criteria, epsilon AND count of iterations is used.
                 )

filterType='median'
filterSize=(21,21)
resize = (3/4, 3/4)

try:
    with open(save_path + video  + '_test' + '.pickle', 'rb') as file:
        with open(save_path + 'last_stop' + video + '.npy', 'r') as stop_file:

            last_stop = np.fromfile(stop_file, dtype=np.uint32)
            print(last_stop)
            grid = pickle.load(file)
        print('No error in filehandling')
except (EOFError, FileNotFoundError) as e:
    print('First time for this video.')
    last_stop = [np.uint32(0)]
    grid = selection.ParameterGrid(dict(hessianThreshold=np.array([10, 50, 100, 200, 500]),
                                        nOctaves=np.array([1, 3, 5]),
                                        nOctaveLayers=np.array([1, 3, 5]),
                                        )
                                    )

    with open(save_path + video  + '_test' + '.pickle', 'wb') as file:
        pickle.dump(grid, file)

video_path = folder + video
print('video_path: ' + video_path)

if last_stop[-1] < len(grid)-1:
    for y in range(last_stop[-1], len(grid)):
        print('Job %s of %s' %(str(y), str(len(grid))))
        kt = tr.keypoint_tracker(video_path, start_frame=start, end_frame=end)
        results = [video_path, [grid[y]], [lk_params], filterType, filterSize, resize]
        start_t = timeit.default_timer()
        # detector, detector_params, avg_lifespan, hist_lifespan, heatmap = kt.run(detector='good', detector_params=params)
        results.extend(kt.run(  detector='surf',
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
        pos_grid = np.uint32(y)
        with open(save_path + video  + '_test' + '.pickle', 'ab') as file:
            with open(save_path + 'last_stop' + video + '.npy', 'w') as stop_file:
                bin_results = pickle.dumps(results)
                file.write(bin_results)
                pos_grid.tofile(stop_file)

print('Done with grid.')
with open(save_path + video  + '_test' + '.pickle', 'rb+') as file:
    pload = pickle.load(file)
    if type(pload)==pd.core.frame.DataFrame:
        raise ValueError('Dataframe already filled')
    results_arr = []
    while 1:
        try:
            pload = pickle.load(file)
            results_arr.append(pload)
        except EOFError:
            break
# print(results_arr)
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
print('Filling Dataframe...')
print(len(grid))
print('resulsts arr: ')
print(len(results_arr))
for x in range(len(results_arr)):
    for z in range(len(results_arr[x])):
        all_data.iloc[x,z] = results_arr[x][z]
print('sorting by avg_lifespan...')
sorted_data = all_data.sort_values(by='avg_lifespan', ascending=False)
print('saving to: '+ save_path + video  + '_test' + '.pickle...')
file = open(save_path + video  + '_test' + '.pickle', 'wb')
pickle.dump(sorted_data, file)
file.close()
print('done')

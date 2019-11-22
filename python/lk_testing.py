#!/usr/bin/env python
'''
Script to test Lucas-Kanade optical flow. Uses tracking.py
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

folder = '../videos/'
save_path = '../results/no_blinks/dataframes/lk/'

start = 0
end = 10

filterType='median'
filterSize=(21,21)
resize = (3/4, 3/4)


# Try to open the parameter-grid from pickle-file and get last location in it from numpy-file.
# If it fails, create own grid and save it as .pickle
try:
    with open(save_path + video  + '_test' + '.pickle', 'rb') as file:
        with open(save_path + 'last_stop' + video + '.npy', 'r') as stop_file:

            last_stop = np.fromfile(stop_file, dtype=np.uint32)
            print(last_stop)
            grid_lk = pickle.load(file)
        print('No error in filehandling')
except (EOFError, FileNotFoundError) as e:
    print('First time for this video.')
    last_stop = [np.uint32(0)]
    grid_lk = selection.ParameterGrid(dict( winSize=[(5,5), (15, 15), (31,31)],
                                            maxLevel=[0, 3, 6],
                                            criteria=[(3, 4, 0.5),
                                                      (3, 10, 0.03),
                                                      (3, 30, 0.03),
                                                      ],
                                            # termination criteria, epsilon AND count of iterations is used.
                                            detec=[('sift', {'sigma': 2.5, 'nOctaveLayers': 3, 'edgeThreshold': 2, 'contrastThreshold': 0.1}),
                                                ('sift', {'sigma': 2.5, 'nOctaveLayers': 5, 'edgeThreshold': 2, 'contrastThreshold': 0.1}),
                                                ('sift', {'sigma': 1.6, 'nOctaveLayers': 3, 'edgeThreshold': 2, 'contrastThreshold': 0.1}),

                                                ('surf',{'nOctaves':0 , 'nOctaveLayers': 0, 'hessianThreshold': 0}),
                                                ('surf',{'nOctaves':0 , 'nOctaveLayers': 0, 'hessianThreshold': 0}),
                                                ('surf',{'nOctaves':0 , 'nOctaveLayers': 0, 'hessianThreshold': 0}),

                                                ('good',{'qualityLevel': 0.5, 'minDistance': 61, 'maxCorners': 510, 'blockSize': 5}),
                                                ('good',{'qualityLevel': 0.5, 'minDistance': 61, 'maxCorners': 10, 'blockSize': 5})
                                                    ]
                                          )
                                      )
    print(len([p for p in grid_lk]))
    with open(save_path + video  + '_test' + '.pickle', 'wb') as file:
        pickle.dump(grid_lk, file)

video_path = folder + video
print('video_path: ' + video_path)

# Start feature tracker in tracking.py with the last setting from the numpy file
# append results to .pickle file.

if last_stop[-1] < len(grid_lk)-1:
    for y in range(last_stop[-1], len(grid_lk)):
        print('Job %s of %s' %(str(y), str(len(grid_lk))))
        kt = tr.keypoint_tracker(video_path, start_frame=start, end_frame=end)
        print(type(grid_lk))
        print(type(grid_lk[y]))
        print(grid_lk[y])
        grid_dic = grid_lk[y]#.pop('detec')
        detector = grid_dic.pop('detec')
        print('detector')
        print(detector)
        print('gridy')
        print(grid_dic)
        print('gridyende')

        results = [video_path, [detector], [grid_dic], filterType, filterSize, resize]
        start_t = timeit.default_timer()
        # detector, detector_params, avg_lifespan, hist_lifespan, heatmap = kt.run(detector='good', detector_params=params)
        results.extend(kt.run(  detector=detector[0],
                                detector_params=detector[1],
                                lk_params=grid_dic,
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

print('Done with grid_lk.')

# When done with all parameter settings, get all results from .pickle file, build Dataframe and in same file.
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
print(len(grid_lk))
print('resulsts arr: ')
print(len(results_arr))
for x in range(len(results_arr)):
    for z in range(len(results_arr[x])):
        all_data.iloc[x,z] = results_arr[x][z]
print('sorting by avg_lifespan...')
sorted_data = all_data.sort_values(by='avg_lifespan', ascending=False)
print('saving to: '+ save_path + video  + '_test' + '.pickle ...')
file = open(save_path + video  + '_test' + '.pickle', 'wb')
pickle.dump(sorted_data, file)
file.close()
print('done')

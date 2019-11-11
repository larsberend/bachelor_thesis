#!/usr/bin/env python
'''
Script to test shi-tomasi feature detector located in tracking.py
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
save_path = 'results/no_blinks/dataframes/prep/'

start = 0
end = 10


try:
    with open(save_path + video  + '_test' + '.pickle', 'rb') as file:
        with open(save_path + 'last_stop' + video + '.npy', 'r') as stop_file:

            last_stop = np.fromfile(stop_file, dtype=np.uint32)
            print(last_stop)
            grid_prep = pickle.load(file)
        print('No error in filehandling')
except (EOFError, FileNotFoundError) as e:
    print('First time for this video.')
    last_stop = [np.uint32(0)]
    grid_prep = selection.ParameterGrid(dict( resize = [(1/2,1/2),(3/4,3/4),(1,1)],
                                              filterType = ['median', 'gauss'],
                                              filterSize = [(5,5), (15,15), (21,21)],

                                            detec=[('sift', {'sigma': 1.0, 'nOctaveLayers': 5, 'edgeThreshold': 10, 'contrastThreshold': 0.01}),

                                                ('surf',{'nOctaves': 1 , 'nOctaveLayers': 5, 'hessianThreshold': 100}),

                                                ('good',{'qualityLevel': 0.5, 'minDistance': 1, 'maxCorners': 260, 'blockSize': 2}),
                                                    ]
                                          )
                                      )
    print(len([p for p in grid_prep]))
    print(grid_prep[1])
    with open(save_path + video  + '_test' + '.pickle', 'wb') as file:
        pickle.dump(grid_prep, file)

video_path = folder + video
print('video_path: ' + video_path)

if last_stop[-1] < len(grid_prep)-1:
    for y in range(last_stop[-1], len(grid_prep)):
        print('Job %s of %s' %(str(y), str(len(grid_prep))))
        kt = tr.keypoint_tracker(video_path, start_frame=start, end_frame=end)
        print(type(grid_prep))
        # print(type(grid_prep[y]))
        print(grid_prep[y])
        grid_dic = grid_prep[y]#.pop('detec')
        detector = grid_dic.pop('detec')
        if detector[0] == 'good':
            lk_params = dict(winSize=(15, 15),
                         maxLevel=3,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                         # termination criteria, epsilon AND count of iterations is used.
                         )
        elif detector[0] == 'surf':
            lk_params = dict(winSize=(31, 31),
                         maxLevel=3,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                         # termination criteria, epsilon AND count of iterations is used.
                         )

        elif detector[0] == 'sift':
            lk_params = dict(winSize=(15, 15),
                         maxLevel=0,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 4, 0.5),
                         # termination criteria, epsilon AND count of iterations is used.
                         )


        results = [video_path, [detector], [lk_params], [grid_dic]]
        start_t = timeit.default_timer()
        # detector, detector_params, avg_lifespan, hist_lifespan, heatmap = kt.run(detector='good', detector_params=params)
        results.extend(kt.run(  detector=detector[0],
                                detector_params=detector[1],
                                lk_params=lk_params,
                                resize=grid_prep[y]['resize'],
                                filterType=grid_prep[y]['filterType'],
                                filterSize=grid_prep[y]['filterSize'],
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

print('Done with grid_prep.')
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
print(len(grid_prep))
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

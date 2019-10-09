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
import sklearn.model_selection as selection
import timeit
import os

import tracking as tr

folder = '/home/laars/uni/BA/eyes/Gold_Standard_Kopien/videos/'
save_path = '/home/laars/uni/BA/code/python/results/no_blinks/dataframes/'

start = 450
end = 550
lk_params = tr.lk_params
filterType='median'
filterSize=(15,15)
resize = (3/4, 3/4)

grid = selection.ParameterGrid(dict(maxCorners=np.arange(10, 450, 50),
                                    qualityLevel=np.arange(0.1, 1, 0.1),
                                    minDistance=np.arange(1, 51, 10),
                                    blockSize=np.arange(1, 31, 5)
                                    )
                                )
all_data = pd.DataFrame(index = ['detector_params',
                                'video_path',
                                'lk_params',
                                'filterType',
                                'filterSize',
                                'resize',
                                'avg_lifespan',
                                'hist_lifespan',
                                'heatmap',
                                'none_idx',
                                'runtime'
                                ],
                        columns = range(len(grid))
                        )
print(all_data)

# all_data = {'detector_params':[],
#             'video_path':[],
#             'start_frame':[],
#             'end_frame':[],
#             'avg_lifespan':[],
#             'hist_lifespan':[],
#             'heatmap':[],
#             'runtime':[]
#             }
for video in os.listdir(folder):
    video_path = folder + video
    print(video_path)

    for y in range(len(grid)):
        kt = tr.keypoint_tracker(video_path, start_frame=start, end_frame=end)
        results = [[grid[y]], video_path, [lk_params], filterType, filterSize, resize]
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
        print(video_path)
        print(end_t-start_t)
        for x in range(len(results)):
            all_data.iloc[x,y] = results[x]
        break
    break
    # data = {
    #         'detector_params' : [params],
    #         'video_path' : [video_path],
    #         'start_frame': [start],
    #         'end_frame': [end],
    #         'avg_lifespan' : [avg_lifespan],
    #         'hist_lifespan' : [hist_lifespan],
    #         'heatmap' : [heatmap],
    #         'hod' : [hod]
    #         }
    #
    # for ind in data.keys():
    #     all_data[ind].append(data[ind])
print('here 2')
# res_df = pd.DataFrame(all_data) #,index=['detector_params', 'video_path', 'end_frame', 'avg_lifespan', 'hist_lifespan', 'heatmap', 'hod'])
all_data.to_csv(save_path + '_test' + '.csv')
print('done')
print(all_data)

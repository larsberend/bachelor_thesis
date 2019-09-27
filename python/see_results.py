#!/usr/bin/env python
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

data = pd.read_csv('/home/laars/uni/BA/code/python/results/dataframes/good_proto.csv')
print(data)
# print(data.iloc[0]['hist_lifespan'].shape)
hist_lifespan = data.iloc[0]['hist_lifespan']

hist_lifespan = np.asarray(hist_lifespan)
print(hist_lifespan.shape())

plt.hist(hist_lifespan,bins=int(max(hist_lifespan) / 20))
plt.show()

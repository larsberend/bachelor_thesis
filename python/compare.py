#!/usr/bin/env python
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import sys
import bz2


# make pickel files with all files from videos with 120 Hz, saving as pickle
def make_comp(detector):
    comp = [[],[],[],[],[],[],[]]
    for i in range(10):
        for k in range(2):
            file = bz2.open('results/no_blinks/dataframes/%s/%s_eye%s.mp4_test.pickle.bz2'%(detector, i,k), 'rb')
            data = pickle.load(file).sort_index()
            file.close()
            comp[0]=list(data.index)
            print(i)
            print(k)
            comp[1].append(data['detector_params'])
            comp[2].append(data['avg_lifespan'])
            comp[3].append(data['runtime'])
            comp[4].append(data['lk_params'])
            comp[5].append(data['hist_lifespan'])
            comp[6].append(data['preprocessing'])
    with open('results/no_blinks/dataframes/%s/comp120.pickle'%(detector), 'wb') as file:
        pickle.dump(comp, file)

# make dataframes for videos with 200 Hz, saving as pickle
def make_comp200(detector):
    comp = [[],[],[],[],[],[],[]]
    file = bz2.open('results/no_blinks/dataframes/%s/%s_eye%s.mp4_test.pickle.bz2'%(detector, 200,0), 'rb')
    data = pickle.load(file).sort_index()
    file.close()
    comp[0]=list(data.index)
    comp[1].append(data['detector_params'])
    comp[2].append(data['avg_lifespan'])
    comp[3].append(data['runtime'])
    comp[4].append(data['lk_params'])
    comp[5].append(data['hist_lifespan'])
    comp[6].append(data['preprocessing'])
    file = bz2.open('results/no_blinks/dataframes/%s/%s_eye%s.mp4_test.pickle.bz2'%(detector, 200,1), 'rb')
    data = pickle.load(file).sort_index()
    file.close()
    comp[2].append(data['avg_lifespan'])
    comp[3].append(data['runtime'])
    comp[5].append(data['hist_lifespan'])
    comp[6].append(data['preprocessing'])
    with open('results/no_blinks/dataframes/%s/comp200.pickle'%(detector), 'wb') as file:
        pickle.dump(comp, file)

# take pickled dataframes and create pretty csv files.

def see_comp(detector, with_200):
    if with_200==True:
        with open('results/no_blinks/dataframes/%s/comp200.pickle'%(detector), 'rb') as file:
            comp=pickle.load(file)
    else:
        with open('results/no_blinks/dataframes/%s/comp120.pickle'%(detector), 'rb') as file:
            comp=pickle.load(file)
    for ls in comp[2][1:]:
        comp[2][0] += ls
    comp[2][0][:] = [x / (2 if with_200 else 20) for x in comp[2][0]]
    for t in comp[3][1:]:
        comp[3][0] += t
    comp[3][0][:] = [x / (2 if with_200 else 20) for x in comp[3][0]]
    # print(comp[5][0])
    for length in range(len(comp[5][0])):
            if type(comp[5][0][length]) is np.ndarray:
                comp[5][0][length] = len(comp[5][0][length])
            else:
                comp[5][0][length] = 0
    for hist in comp[5][1:]:
        for l0, l1 in zip(range(len(comp[5][0])), hist):
            if comp[5][0][l0] is not None:
                if type(l1) is np.ndarray:
                    comp[5][0][l0] += len(l1)
    comp[5][0][:] = [x / (2 if with_200 else 20) for x in comp[5][0]]


    #print(comp[1][0])
    comp = pd.DataFrame([comp[3][0],comp[2][0],comp[1][0],comp[4][0], comp[6][0], comp[5][0]]).transpose()#, index=comp[0])
    # comp = pd.DataFrame([comp[3][0],comp[2][0],comp[1][0],comp[4][0], comp[5][0]]).transpose()#, index=comp[0])
    comp = comp.sort_values(by='avg_lifespan', ascending=False)
    if with_200:
        with open('results/no_blinks/dataframes/%s/comp200.csv'%(detector), 'w') as file:
            comp.to_csv(file)
    else:
        with open('results/no_blinks/dataframes/%s/comp120.csv'%(detector), 'w') as file:
            comp.to_csv(file)
    print('Ende')


if __name__=='__main__':
    # make_comp('gfeat')
    # make_comp('dog')
    # make_comp('fhess')
    # make_comp('lk')
    # make_comp('lk_sift')
    # make_comp('prep_san')
    make_comp('prep')
    #
    # make_comp200('fhess')
    # make_comp200('gfeat')
    # make_comp200('dog')
    #
    # make_comp200('prep200')
    # make_comp200('prep')
    # make_comp200('prep200_san')
    # see_comp('dog', True)
    # see_comp('fhess', True)
    # see_comp('gfeat', True)
    # see_comp('lk', True)
    # see_comp('prep200_san', True)
    # see_comp('prep200', True)
    # print('ohne 200\n\n\n')
    #
    # see_comp('dog',False)
    # see_comp('fhess', False)
    # see_comp('gfeat', False)
    # see_comp('lk', False)
    # see_comp('lk_sift', False)
    see_comp('prep', False)

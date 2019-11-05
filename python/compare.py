#!/usr/bin/env python
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import sys
import bz2

def make_comp(detector):
    comp = [[],[],[],[]]
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
    with open('results/no_blinks/dataframes/%s/comp120.pickle'%(detector), 'wb') as file:
        pickle.dump(comp, file)

def make_comp200(detector):
    comp = [[],[],[],[]]
    file = bz2.open('results/no_blinks/dataframes/%s/%s_eye%s.mp4_test.pickle.bz2'%(detector, 200,0), 'rb')
    data = pickle.load(file).sort_index()
    file.close()
    comp[0]=list(data.index)
    comp[1].append(data['detector_params'])
    comp[2].append(data['avg_lifespan'])
    comp[3].append(data['runtime'])
    file = bz2.open('results/no_blinks/dataframes/%s/%s_eye%s.mp4_test.pickle.bz2'%(detector, 200,1), 'rb')
    data = pickle.load(file).sort_index()
    file.close()
    comp[2].append(data['avg_lifespan'])
    comp[3].append(data['runtime'])
    with open('results/no_blinks/dataframes/%s/comp200.pickle'%(detector), 'wb') as file:
        pickle.dump(comp, file)



def see_comp(detector, with_200):
    if with_200==True:
        with open('results/no_blinks/dataframes/%s/comp200.pickle'%(detector), 'rb') as file:
            comp=pickle.load(file)
    else:
        with open('results/no_blinks/dataframes/%s/comp120.pickle'%(detector), 'rb') as file:
            comp=pickle.load(file)
    for ls in comp[2][1:]:
        comp[2][0] += ls
    for t in comp[3][1:]:
        comp[3][0] += t
    #print(comp[1][0])
    comp = pd.DataFrame([comp[3][0],comp[2][0],comp[1][0]]).transpose()#, index=comp[0])
    comp = comp.sort_values(by='avg_lifespan', ascending=False)
    if with_200:
        with open('results/no_blinks/dataframes/%s/comp200.csv'%(detector), 'w') as file:
            comp.to_csv(file)
    else:
        with open('results/no_blinks/dataframes/%s/comp.csv'%(detector), 'w') as file:
            comp.to_csv(file)
    print('Ende')


if __name__=='__main__':
    make_comp('fhess')
    make_comp('gfeat')
    make_comp('dog')

    make_comp200('fhess')
    make_comp200('gfeat')
    make_comp200('dog')
    see_comp('dog', True)
    see_comp('fhess', True)
    see_comp('gfeat', True)
    print('ohne 200\n\n\n')
    see_comp('dog',False)
    see_comp('fhess', False)
    see_comp('gfeat', False)

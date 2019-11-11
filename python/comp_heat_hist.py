#!/usr/bin/env python
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import sys
import bz2

def make_comp(detector):
    comp = [[],[],[]]
    for i in range(10):
        for k in range(2):
            if(detector=='gfeat' and i==0 and k == 0):
                continue
            file = bz2.open('results/no_blinks/dataframes/%s/%s_eye%s.mp4_test.pickle.bz2'%(detector, i,k), 'rb')
            data = pickle.load(file).sort_index()
            file.close()
            comp[0]=list(data.index)
            print(i)
            print(k)
            comp[1].append(data['heatmap'])
            comp[2].append(data['hist_lifespan'])
    with open('results/no_blinks/dataframes/%s/comp_heat_hist120.pickle'%(detector), 'wb') as file:
        print('dumping...')
        pickle.dump(comp, file)
        print('done')

def see_comp(detector):
    with open('results/no_blinks/dataframes/%s/comp_heat_hist120.pickle'%(detector), 'rb') as file:
        comp=pickle.load(file)
    for heat in comp[1][1:]:
        comp[1][0] += heat
    for hist in comp[2][1:]:
        for l0, l1 in zip(range(len(comp[2][0])), hist):
            if comp[2][0][l0] is not None:
                if l1 is not None:
                    comp[2][0][l0] = np.concatenate((comp[2][0][l0], l1))
    with open('results/no_blinks/dataframes/%s/comp120.pickle'%(detector), 'rb') as file:
        comp_life=pickle.load(file)
    for ls in comp_life[2][1:]:
        comp_life[2][0] += ls
    for t in comp_life[3][1:]:
        comp_life[3][0] += t
    comp = pd.DataFrame([comp[2][0], comp[1][0], comp_life[2][0], comp_life[1][0],comp_life[4][0]]).transpose()
    comp = comp.sort_values(by='avg_lifespan', ascending=False)

    # for heat, ls, para, lk_para in zip(comp['heatmap'], comp['avg_lifespan'], comp['detector_params'], comp['lk_params']):
        # heatmap_norm = cv.normalize(heat, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U).transpose(1, 0)
        # # heatmapshow = cv.applyColorMap(heatmap_norm,  cv.COLORMAP_CIVIDIS)
        # # cv.imshow('results/no_blinks/dataframes/%s/heat/'%(detector) + str(ls) + '_' + str(para) + '.png', heatmapshow)
        # # cv.waitKey(0)
        # cv.imwrite('results/no_blinks/dataframes/%s/heat/inv_'%(detector) + str(ls) + '_' + str(lk_para) + '_' + str(para) + '.png', cv.bitwise_not(heatmap_norm))
        # print(ls)
        # print(para)
        # print(lk_para)
    #
    for hist, ls, para, lk_para in zip( comp['hist_lifespan'], comp['avg_lifespan'], comp['detector_params'], comp['lk_params']):
        print(ls)
        print(para)
        print(lk_para)
        print(len(hist))
        plt.hist(np.array(hist) ,bins=int(max(hist) / 10))
        plt.show()

            # for heat in data.head(1)['heatmap']:
            #     # print(heat)
            #     if k%2 == 0:
            #         print('aha')
            #         heat = cv.flip(heat, 1)






if __name__=='__main__':
    # make_comp('lk')
    # see_comp('lk')
    make_comp('fhess')
    make_comp('dog')

    see_comp('fhess',False)
    see_comp('dog', False)
    # see_comp('fhess')
    # see_comp('gfeat')
    # see_comp('dog')

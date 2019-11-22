#!/usr/bin/env python
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import sys
import bz2
from matplotlib import cm



# make and save either histograms or heatmaps and save in folders corresponding to Nr. in parameter-grid
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
            # for ind in [8]:
            #     heat = data.ix[ind]['heatmap'].T
            #     # heat = cv.normalize(heat.T, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            #     # heat = cv.applyColorMap(heat,  cv.COLORMAP_OCEAN)
            #     cv.imwrite('results/no_blinks/dataframes/%s/heat/%s/%s_eye%s.png'%(detector, ind, i, k), heat)
            for ind in [8]:
                hist = data.ix[ind]['hist_lifespan']
                fig = plt.figure()
                ax = plt.subplot(111, xlabel='Lifespan in frames', ylabel='Nr. of tracks')
                ax.hist(hist ,bins=int(max(hist) / 10))
                # plt.show()
                # ax.xlabel('Lifespan in frames')
                # ax.ylabel('Nr. of tracks')
                fig.savefig('results/no_blinks/dataframes/%s/hist/%s/%s_eye%s.png'%(detector, ind, i, k))
                plt.close(fig)

    # with open('results/no_blinks/dataframes/%s/comp_heat_hist.pickle'%(detector), 'wb') as file:
    #     print('dumping...')
    #     pickle.dump(comp, file)
    #     print('done')


# save histograms and heatmaps in a dataframe, not much used.
def see_comp(detector):
    with open('results/no_blinks/dataframes/%s/comp_heat_hist.pickle'%(detector), 'rb') as file:
        comp=pickle.load(file)
    # for x in range(len(comp[1][1:])):
    #     print(x)
    #     comp[1][0].append(comp[1][x])
    #     print(comp[1][0])
    heat = np.array(comp[1])
    for y in range(len(comp[2][1:])):
        for l0, l1 in zip(range(len(comp[2][0])), comp[2][y]):
            if comp[2][0][l0] is not None:
                if l1 is not None:
                    comp[2][0][l0] = np.concatenate((comp[2][0][l0], l1))


    with open('results/no_blinks/dataframes/%s/comp120.pickle'%(detector), 'rb') as file:
        comp_life=pickle.load(file)
    for ls in comp_life[2][1:]:
        comp_life[2][0] += ls
    for t in comp_life[3][1:]:
        comp_life[3][0] += t
    comp = pd.DataFrame([comp[2][0], comp[1], comp_life[2][0], comp_life[1][0],comp_life[4][0]]).transpose()
    comp = comp.sort_values(by='avg_lifespan', ascending=False)
    print(comp.index)
    heat = comp.iloc[48]['heatmap']
    print(heat.shape)
    '''
    for heat, ls, para, lk_para in zip(comp['heatmap'], comp['avg_lifespan'], comp['detector_params'], comp['lk_params']):
        heatmap_norm = cv.normalize(heat, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U).transpose(1, 0)
        heatmapshow = cv.applyColorMap(heatmap_norm,  cv.COLORMAP_CIVIDIS)
        cv.imshow('results/no_blinks/dataframes/%s/heat/'%(detector) + str(ls) + '_' + str(para) + '.png', heatmapshow)
        cv.waitKey(0)
        # cv.imwrite('results/no_blinks/dataframes/%s/heat/inv_'%(detector) + str(ls) + '_' + str(lk_para) + '_' + str(para) + '.png', cv.bitwise_not(heatmap_norm))
        print(ls)
        print(para)
        print(lk_para)
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
    '''





if __name__=='__main__':
    # see_comp('prep')
    make_comp('prep')



    # make_comp('lk')
    # see_comp('lk')
    comp = [[],[],[]]
    detector = 'prep200'
    for k in range(2):
        # file = bz2.open('results/no_blinks/dataframes/%s/%s_eye%s.mp4_test.pickle.bz2'%(detector, 200, k), 'rb')
        data = pickle.load(file).sort_index()
        file.close()
        comp[0]=list(data.index)
        print(k)
        comp[1].append(data['heatmap'])
        comp[2].append(data['hist_lifespan'])
        # for ind in [7, 41, 42]:
        #     heat = data.ix[ind]['heatmap']
        #     heat = cv.normalize(heat.T, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        #     heat = cv.applyColorMap(heat,  cv.COLORMAP_OCEAN)
        #     cv.imwrite('results/no_blinks/dataframes/%s/heat/%s/%s_eye%s.png'%(detector, ind, 200, k), heat)
        for ind in [18, 41, 45]:
            hist = data.ix[ind]['hist_lifespan']
            fig = plt.figure()
            ax = plt.subplot(111, xlabel='Lifespan in frames', ylabel='Nr. of tracks')
            ax.hist(hist ,bins=int(max(hist) / 10))
            # plt.show()
            # ax.xlabel('Lifespan in frames')
            # ax.ylabel('Nr. of tracks')
            fig.savefig('results/no_blinks/dataframes/%s/hist/%s/%s_eye%s.png'%(detector, ind, 200, k))
            plt.close(fig)
    # with open('results/no_blinks/dataframes/%s/comp_heat_hist.pickle'%(detector), 'wb') as file:
    #     print('dumping...')
    #     pickle.dump(comp, file)
    #     print('done')

    # make_comp('dog')

    # see_comp('fhess',False)
    # see_comp('fhess')
    # see_comp('gfeat')
    # see_comp('dog')

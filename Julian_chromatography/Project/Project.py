#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:50:36 2019

@author: ilia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter
from peakutils import baseline
import easygui

class Chromo_clusterize():
    
    def __init__(self):
        self.Dataset = []
        self.picks = []
        self.Subset = []
        self.Subset_filtered = []
        self.Subset_debaselined = []
    
    def load_data(self):
        msg = 'Please choose dataset location'
        path = easygui.fileopenbox(msg=msg)
        file_content = pd.read_excel(path, header=None)
        self.Dataset = file_content
        print('Dataset loaded')
        
    def select_region(self, layer_to_display=2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Please select subset')
        line, = ax.plot(np.array(self.Dataset.iloc[1:, layer_to_display], 
                                 dtype='float'), picker=1) # 5 points tolerance
        picks = []
        def onpick(event):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ind = event.ind
            print(xdata[ind][1])
            picks.append(xdata[ind][1])
            if len(picks) == 2:
                plt.close(fig)
                print('Region selected!')
        self.picks = picks
        fig.canvas.mpl_connect('pick_event', onpick)
        
        
    def generate_subset(self):
        if len(self.picks) == 2:
            self.Subset = self.Dataset.iloc[int(self.picks[0]):int(self.picks[1]), :]
            print('Subset created!')
        else:
            print('Subset not created')
            
    def sample_plot(self, Set='Subset', start_layer=1):
        if Set == 'Subset':
            Subset = np.array(self.Subset, dtype='float')
        elif Set == 'Subset_filtered':
            Subset = np.array(self.Subset_filtered, dtype='float')
        elif Set == 'Subset_debaselined':
            Subset = np.array(self.Subset_debaselined, dtype='float')
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        def cc(arg):
            return mcolors.to_rgba(arg, alpha=0.6)
        
        xs = Subset[:, 0]
        verts = []
        zs = [0.0, 1.0, 2.0, 3.0]
        for z in zs:
            ys = Subset[:, int(z+start_layer)]
            ys[0], ys[-1] = 0, 0
            verts.append(list(zip(xs, ys)))
        
        poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
                                                 cc('y')])
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs, zdir='y')
        ax.set_xlabel('X: Time(seconds)')
        ax.set_xlim3d(min(Subset[:, 0]), max(Subset[:, 0]))
        ax.set_ylabel('Y: No_Series')
        ax.set_ylim3d(-1, 4)
        ax.set_zlabel('Z: Intensity')
        ax.set_zlim3d(0, np.amax(Subset))
        
   
    def filter_subset(self, polyorder=2, window=21):
        self.Subset_filtered = savgol_filter(self.Subset, window_length=window, 
                                             polyorder=polyorder, axis = 0)
        print('Subset filtered!')
    
    def debaseline_subset(self, poly_order=1):
        Subset_filtered = self.Subset_filtered
        Debaselined = np.zeros(np.shape(Subset_filtered))
        Debaselined[:, 0] = Subset_filtered[:, 0]
        for i in range(1, np.shape(Debaselined)[1]):
            vect = Subset_filtered[:, i]
            base = baseline(vect, poly_order)
            vect_debaselined = vect - base
            Debaselined[:, i] = vect_debaselined
        self.Subset_debaselined = Debaselined[1:-1, :]
        print('Subset debaselined!')


Chr = Chromo_clusterize()
Chr.load_data()
Chr.select_region()
Chr.generate_subset()
Chr.sample_plot(Set='Subset', start_layer=5)
Chr.filter_subset(window=51)
Chr.sample_plot(Set='Subset_filtered', start_layer=5)
Chr.debaseline_subset(poly_order=1)
Chr.sample_plot(Set='Subset_debaselined', start_layer=5)






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
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.linear_model import RANSACRegressor
import easygui
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
#from bayes_opt.util import load_logs
#import json
from Dixon import Dixon_test


class Chromo_clusterize():
    
    def __init__(self):
        self.Dataset = []
        self.picks = []
        self.Subset = []
        self.Subset_filtered = []
        self.Subset_debaselined = []
        self.Peak_properties_list = []
        self.cluster = []
        self.cluster_inertia = []
        self.optimizer = []
        self.max_params = []
        self.inertia_opt_progress = []
    
    
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
    
    
    def debaseline_subset(self, Set='Subset_filtered', 
                          poly_order=1):
        if Set == 'Subset_filtered':
            Subset_filtered = self.Subset_filtered
        if Set == 'Subset':
            Subset_filtered = self.Subset
        Debaselined = np.zeros(np.shape(Subset_filtered))
        Debaselined[:, 0] = Subset_filtered[:, 0]
        for i in range(1, np.shape(Debaselined)[1]):
            vect = Subset_filtered[:, i]
            base = baseline(vect, poly_order)
            vect_debaselined = vect - base
            Debaselined[:, i] = vect_debaselined
        self.Subset_debaselined = Debaselined[1:-1, :]
        print('Subset debaselined!')
        
        
    def extract_peaks(self, Set='Subset_debaselined',
                      prominence=10e4, width=20):
        Peak_properties_list = pd.DataFrame([])
        if Set == 'Subset_debaselined':
            Subset_debaselined = self.Subset_debaselined
        elif Set == 'Subset_filtered':
            Subset_debaselined = self.Subset_filtered
        elif Set == 'Subset':
            Subset_debaselined = self.Subset
        for i in range(1, np.shape(Subset_debaselined)[1]):
            vect = Subset_debaselined[:, i]
            peaks, properties = find_peaks(vect, prominence=10e4, width=20)
            properties['peaks'] = peaks
            properties['Sample_number'] = i
            properties = pd.DataFrame(properties)    
            Peak_properties_list = Peak_properties_list.append(properties)
        Peak_properties_list.index = range(0, len(Peak_properties_list))
        self.Peak_properties_list = Peak_properties_list
        print('Peaks extracted!')
        
        
    def plot_extracted_peaks(self, layer=1):
        plt.figure()
        vect = self.Subset_debaselined[:, layer]
        peaks, properties = find_peaks(vect, prominence=10e4, width=10)
        plt.plot(vect)
        plt.plot(peaks, vect[peaks], "x")
        plt.vlines(x=peaks, ymin=vect[peaks] - properties["prominences"],
                   ymax = vect[peaks], color = "C1")
        plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], 
                   xmax=properties["right_ips"], color = "C1")
        
        
    def opti_clust(self):
        Pr = Chr.Peak_properties_list
        Pr_sample = scale(Pr.iloc[:, [0,7]])
        inertia_list = np.array([])
        for i in range(1, 25):
            kmeans = KMeans(n_clusters=i, n_init = 50)
            kmeans = kmeans.fit(Pr_sample)
            inertia_list = np.append(inertia_list, kmeans.inertia_)
            
            if i >= 3:
                y = inertia_list[:i]
                X = np.array([i for i in range(len(np.ones_like(y)))], 
                              dtype='float')
                X = X.reshape(1,-1).T
                reg = RANSACRegressor().fit(X, y)
                y_predicted = reg.predict(X)
                residuals = y - y_predicted
                outliers = Dixon_test().dixon_test(residuals, left = False, 
                                     q_dict='Q90')
                if outliers[1] is not None:
                    print('Stop! Optimal num_clust is: ', i)
                    break
        num_clust = i
        kmeans = KMeans(n_clusters=num_clust, n_init = 100)
        self.cluster = kmeans.fit_predict(Pr_sample)
        self.cluster_inertia = kmeans.fit(Pr_sample).inertia_ / np.shape(Pr_sample)[0]
            
        
    def optimize_prepare_data(self, poly_debase=1, polyorder=2, prominence=10e4,
                               width=20, window=21):
        polyorder = int(polyorder)
        window = int(window)
        poly_debase = int(poly_debase)
        prominence = int(prominence)
        width = int(width)
        def round_up_to_odd(f):
            f = int(np.ceil(f))
            return f + 1 if f % 2 == 0 else f
        window = round_up_to_odd(window)
        self.generate_subset()
        self.filter_subset(polyorder=polyorder, window=window)
        self.debaseline_subset(poly_order=poly_debase)
        self.extract_peaks(prominence=prominence, width=width)
        self.opti_clust()
        self.inertia_opt_progress.append(self.cluster_inertia)
        return(-self.cluster_inertia)
        
    def run_optimization(self, init_points=2, n_iter=30):
        pbounds = {'polyorder': (1, 4), 'window': (5, 200),
                   'poly_debase':(1, 10), 'prominence':(10e3, 10e6),
                   'width':(1, 200)}
        optimizer = BayesianOptimization(
                f=self.optimize_prepare_data,
                pbounds=pbounds,
                random_state=1)
        logger = JSONLogger(path="./logs.json")
        optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
        optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter)
        self.optimizer = optimizer
        self.max_params = optimizer.max
        print('Optimization complete!')




Chr = Chromo_clusterize()
Chr.load_data()
Chr.select_region()
Chr.generate_subset()
Chr.filter_subset(window=51)
Chr.debaseline_subset(poly_order=1)
Chr.extract_peaks(Set='Subset_debaselined')
Chr.plot_extracted_peaks(layer=1)
Chr.opti_clust()


Chr.sample_plot(Set='Subset_debaselined', start_layer=1)
Chr.sample_plot(Set='Subset', start_layer=1)
Chr.sample_plot(Set='Subset_filtered', start_layer=1)

Chr.run_optimization(init_points=50, n_iter=200)
plt.figure()
plt.plot(Chr.inertia_opt_progress)

Chr.optimize_prepare_data(*Chr.max_params['params'].values())









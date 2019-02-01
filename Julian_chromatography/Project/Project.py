#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:50:36 2019

@author: ilia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
#from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter
from peakutils import baseline
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
#from sklearn.linear_model import RANSACRegressor
import easygui
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
#from bayes_opt.util import load_logs
#import json
#from skopt import gp_minimize
#from scipy.optimize import minimize
#from skopt.plots import plot_convergence
#from Dixon import Dixon_test
from DBI import db_index

class Chromo_clusterize():
    
    def __init__(self):
        self.Dataset = []
        self.picks = []
        self.Subset = []
        self.Subset_filtered = []
        self.Subset_debaselined = []
        self.Peak_properties_list = []
        self.cluster = []
        self.cluster_quality = []
        self.optimizer = []
        self.max_params = []
        self.quality_opt_progress = []
        self.scikit_optimized = []
    
    
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
            peaks, properties = find_peaks(vect, prominence=prominence, 
                                           width=width)
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
#        peaks, properties = find_peaks(vect, prominence=10e4, width=10)
        properties = self.Peak_properties_list[
                self.Peak_properties_list['Sample_number'] == layer]
        peaks = properties['peaks']
        plt.plot(vect)
        plt.plot(peaks, vect[peaks], "x")
        plt.vlines(x=peaks, ymin=vect[peaks] - properties["prominences"],
                   ymax = vect[peaks], color = "C1")
        plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], 
                   xmax=properties["right_ips"], color = "C1")
        
        
#    def opti_clust(self):
#        Pr = Chr.Peak_properties_list
#        Pr_sample = scale(Pr.iloc[:, [0,7]])
#        inertia_list = np.array([])
#        for i in range(1, 25):
#            kmeans = KMeans(n_clusters=i, n_init = 50)
#            kmeans = kmeans.fit(Pr_sample)
#            inertia_list = np.append(inertia_list, kmeans.inertia_)
#            
#            if i >= 3:
#                y = inertia_list[:i]
#                X = np.array([i for i in range(len(np.ones_like(y)))], 
#                              dtype='float')
#                X = X.reshape(1,-1).T
#                reg = RANSACRegressor().fit(X, y)
#                y_predicted = reg.predict(X)
#                residuals = y - y_predicted
#                outliers = Dixon_test().dixon_test(residuals, left = False, 
#                                     q_dict='Q90')
#                if outliers[1] is not None:
#                    print('Stop! Optimal num_clust is: ', i)
#                    break
#        num_clust = i
#        kmeans = KMeans(n_clusters=num_clust, n_init = 100, 
#                        init='random', max_iter = 1000, tol = 1e-6
#                        )
#        self.cluster = kmeans.fit_predict(Pr_sample)
#        self.cluster_quality = kmeans.fit(Pr_sample).inertia_ / np.shape(Pr_sample)[0]
#            
        
    def opti_clust_DBI(self):
         Pr = Chr.Peak_properties_list
         Pr_sample = scale(Pr.iloc[:, [0, 7]])
         db_list = np.array([])
         n_clusters_list = np.array([])
         max_no_clusters = np.shape(Pr_sample)[0] // 5
         print('Analysing ', max_no_clusters, ' clusters ', 
               'Out of ', np.shape(Pr_sample)[0], ' samples')
         for i in range(2, max_no_clusters):
            kmeans = KMeans(n_clusters=i, n_jobs=-1)
            kmeans = kmeans.fit(Pr_sample)
            kmeans_labels = kmeans.labels_
            db = db_index(Pr_sample, kmeans_labels)
            db_list = np.append(db_list, db)
            n_clusters_list = np.append(n_clusters_list, i)
         index = np.where(db_list == min(db_list))[0][0]
         opt_clust_number = int(n_clusters_list[index])
         kmeans = KMeans(n_clusters=opt_clust_number)
         print('Optimal DBI num_clust is: ', opt_clust_number)
         self.cluster = kmeans.fit_predict(Pr_sample)
         self.cluster_quality = min(db_list)
    
    
    def optimize_prepare_data(self, poly_debase=1, polyorder=2, prominence=10e4,
                               width=20, window=21):
#        poly_debase = int(poly_debase)
#        polyorder = int(polyorder)
#        prominence = int(prominence)
#        width = int(width)
#        window = int(window)
        assert type(poly_debase) == int
        assert type(polyorder) == int
        assert type(prominence) == int
        assert type(width) == int
        assert type(window) == int
        def round_up_to_odd(f):
            f = int(np.ceil(f))
            return f + 1 if f % 2 == 0 else f
        window = round_up_to_odd(window)
        self.generate_subset()
        self.filter_subset(polyorder=polyorder, window=window)
        self.debaseline_subset(poly_order=poly_debase)
        self.extract_peaks(prominence=prominence, width=width)
        self.opti_clust_DBI()
        self.quality_opt_progress.append(self.cluster_quality)
        return(-self.cluster_quality)
        
        
    def function_to_be_optimized(self, poly_debase, polyorder, prominence,
                               width, window):
            poly_debase = int(poly_debase)
            polyorder = int(polyorder)
            prominence = int(prominence)
            width = int(width)
            window = int(window)
            return self.optimize_prepare_data(poly_debase, polyorder, prominence,
                                   width, window)
        
        
    def run_optimization(self, init_points=2, n_iter=30, alpha=1e-5):
        Chr.quality_opt_progress = []
        pbounds = {'polyorder': (2, 2), 'window': (10, 500),
                   'poly_debase':(1, 1), 'prominence':(10e4, 10e4),
                   'width':(10, 10)}
        optimizer = BayesianOptimization(
                f=self.function_to_be_optimized,
                pbounds=pbounds)
        logger = JSONLogger(path="./logs.json")
        optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
        optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter, 
                alpha=alpha)
        self.optimizer = optimizer
        self.max_params = optimizer.max
        print('Optimization complete!')
        
        
#    def run_optimization2_scikit(self):
#        pbounds = [(2, 3)], 
##                   (2, 3),
##                   (10e5, 10e6),
##                   (50, 60),
##                   (10, 30)]
#        res = gp_minimize(
#                func = self.optimize_prepare_data,
#                dimensions=pbounds,
#                n_calls=15, 
#                n_random_starts=5,
#                acq_func='LCB'
#                )
#        plot_convergence(res)
#        self.scikit_optimized = res
        

Chr = Chromo_clusterize()
Chr.load_data()
Chr.select_region()
Chr.picks = [7000, 8400]
Chr.generate_subset()
Chr.filter_subset(window=51)
Chr.debaseline_subset(poly_order=1)
Chr.extract_peaks(Set='Subset_debaselined', prominence=10e3, width=20)


Chr.sample_plot(Set='Subset', start_layer=1)
Chr.sample_plot(Set='Subset_filtered', start_layer=1)
Chr.sample_plot(Set='Subset_debaselined', start_layer=1)
Chr.plot_extracted_peaks(layer=10)

Chr.run_optimization(init_points=5, n_iter=30, alpha=1e-5)
plt.figure()
plt.plot(Chr.quality_opt_progress)

Chr.function_to_be_optimized(*Chr.max_params['params'].values())

#Chr.run_optimization2_scikit()









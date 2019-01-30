import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap

Dataset = pd.read_excel('Lp.xlsx', header=None)
Subset = np.array(Dataset.iloc[3500:4000, 0:], dtype='float')

#Plotting 3d plot of 4 sample series
def sample_plot(Subset):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    def cc(arg):
        return mcolors.to_rgba(arg, alpha=0.6)
    
    xs = Subset[:, 0]
    verts = []
    zs = [0.0, 1.0, 2.0, 3.0]
    for z in zs:
        ys = Subset[:, int(z+1)]
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
    plt.show()

# Signal filtering 
from scipy.signal import savgol_filter
Subset_filtered = savgol_filter(Subset, window_length=21, polyorder=1, 
                                axis = 0)

# Baseline correction
from peakutils import baseline

def debaseline_subset(Subset_filtered, poly_order=1):
    Debaselined = np.zeros(np.shape(Subset_filtered))
    Debaselined[:, 0] = Subset_filtered[:, 0]
    for i in range(1, np.shape(Debaselined)[1]):
        vect = Subset_filtered[:, i]
        base = baseline(vect, poly_order)
        vect_debaselined = vect - base
        Debaselined[:, i] = vect_debaselined
    return Debaselined[1:-1, :]

Subset_debaselined = debaseline_subset(Subset_filtered)

sample_plot(Subset)
sample_plot(Subset_filtered)
sample_plot(Subset_debaselined)

# Peak detection
from scipy.signal import find_peaks

Peak_properties_list = pd.DataFrame([])

for i in range(1, np.shape(Subset_debaselined)[1]):
    vect = Subset_debaselined[:, i]
    peaks, properties = find_peaks(vect, prominence=10e4, width=20)
    properties['peaks'] = peaks
    properties['Sample_number'] = i
    properties = pd.DataFrame(properties)    
    Peak_properties_list = Peak_properties_list.append(properties)
    
Peak_properties_list.index = range(0, len(Peak_properties_list))

plt.figure()
vect = Subset_debaselined[:, 1]
peaks, properties = find_peaks(vect, prominence=10e4, width=10)
plt.plot(vect)
plt.plot(peaks, vect[peaks], "x")
plt.vlines(x=peaks, ymin=vect[peaks] - properties["prominences"],
           ymax = vect[peaks], color = "C1")
plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], 
           xmax=properties["right_ips"], color = "C1")
plt.show()


# Hierarchical clustering of data
#from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage  
from sklearn.cluster import AgglomerativeClustering

linked = linkage(Peak_properties_list.iloc[:, :-1], method='ward', 
                 metric='euclidean')
plt.figure()
dendrogram(linked,
            orientation='top',
            #labels=labelList,
            distance_sort='descending')

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean',
                                  linkage='ward')  
clust_labels = cluster.fit_predict(Peak_properties_list.iloc[:, :-1])




# Automatic picker

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Please select subset')

line, = ax.plot(np.array(Data.iloc[1:, 2], dtype='float'), picker=1) # 5 points tolerance
picks = []

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
#    ydata = thisline.get_ydata()
    ind = event.ind
    print(xdata[ind][1])
    picks.append(xdata[ind][1])
    if len(picks) == 2:
        plt.close()
        print('Subset selected!')
        
#    points = tuple(zip(xdata[ind], ydata[ind]))
#    print('onpick points:', points)
    

fig.canvas.mpl_connect('pick_event', onpick)

# K-means
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

Pr_sample = scale(Pr.iloc[:, [0,7]])
inertia_list = np.array([])
for i in range(1, 30):
    kmeans = KMeans(n_clusters=i)
    kmeans = kmeans.fit(Pr_sample)
    inertia_list = np.append(inertia_list, kmeans.inertia_)
    
plt.figure()
plt.plot(inertia_list)
plt.figure()
plt.plot(np.diff(inertia_list, n=1))
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit_predict(Pr_sample)

# Automatic cluster selection
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor

y = inertia_list[:5]
X = np.array([i for i in range(len(np.ones_like(y)))], dtype='float')
X = X.reshape(1,-1).T
#X = np.vstack((X, y)).T
#reg = TheilSenRegressor().fit(X, y)
#y_predicted = reg.predict(X)
#residuals = y - y_predicted
#plt.figure()
#plt.scatter(X[:, 0], y)
#plt.plot(X[:, 0], y_predicted)
#plt.figure()
#plt.hist(residuals,  bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
#         
reg2 = RANSACRegressor().fit(X, y)
y_predicted2 = reg2.predict(X)
residuals2 = y - y_predicted2
plt.figure()
plt.scatter(X[:, 0], y)
plt.plot(X[:, 0], y_predicted2)











'''
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage  

linked = linkage(Pr_sample, method='ward', 
                 metric='euclidean')
plt.figure()
dendrogram(linked,
            orientation='top',
            #labels=labelList,
            distance_sort='descending')

plt.figure()
plt.plot(Pr_sample)
'''

def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

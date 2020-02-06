import random
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy import linalg
import math
from rotations import rotate_via_numpy


def gauss_2d(mu, sigma):
    x = random.gauss(mu, sigma)
    y = random.gauss(mu, sigma)
    return [x, y]


def plot_one_dataset(data,fig,title,rank_plot=1,nb_plots=3,plot_index=1, lim = (-2.0, 2.0, 0.5)):
    x, y = data[:,0].tolist(),data[:,1].tolist()

    ax = fig.add_subplot(rank_plot, nb_plots, plot_index)
    ax.set_title(title)

    ## Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    #ax.spines['right'].set_color('none')
    #ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')



    plt.setp(ax, xlim=lim[0:2], ylim=lim[0:2])
    plt.scatter(x, y)
    axes = fig.gca()
    axes.set_aspect('equal')
    axes.set_xticks(np.arange(lim[0],lim[1], lim[2]))
    axes.set_yticks(np.arange(lim[0],lim[1],lim[2]))
    plt.grid()
    plt.tight_layout()



def svd(X):
    #https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/
    # Data matrix X, X doesn't need to be 0-centered
    #n, m = X.shape
    # Compute full SVD
    U, Sigma, Vh = linalg.svd(X, full_matrices=False)
    print("shape of :\n\t-Matrix:{}\n\t-U:{}\n\t-S:{}\n\t-V:{}".format(X.shape,U.shape,len(Sigma), Vh.shape))
    print(np.dot(U.T,U))
    print(np.dot(Vh.T,Vh))
    # Transform X with SVD components
    X_svd = np.dot(U, np.diag(Sigma))
    print(X_svd.shape)
    return X_svd


def rotate(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    _xy = np.empty_like(xy)
    for v in range(xy.shape[0]):
        _xy[v] = rotate_via_numpy(xy[v],radians)

    return _xy


'''
original_data=np.empty([200,2])
for index in range(200):
    original_data[index,:] = gauss_2d(0,0.2)

'''


original_data = np.random.multivariate_normal([0,0], [[0.1,0],[0,0.01]], 200)
svd_original = svd(original_data)

rotated_data = rotate(original_data, -math.radians(45))
svd_rotated= svd(rotated_data)

traslated_data = np.matrix([x+[1,1] for x in rotated_data])
svd_translated= svd(traslated_data)


fig = plt.figure()
plot_one_dataset(original_data, fig, 'Original data', rank_plot=2,nb_plots=3,plot_index=1)
plot_one_dataset(rotated_data, fig, 'rotated data', rank_plot=2,nb_plots=3,plot_index=2)
plot_one_dataset(traslated_data, fig, 'translated data', rank_plot=2,nb_plots=3,plot_index=3)

lim =(-2, 2, 0.5) #(-0.5,0.5,0.1)
plot_one_dataset(svd_original, fig, 'U_original', rank_plot=2,nb_plots=3,plot_index=4,lim=lim)
plot_one_dataset(svd_rotated, fig, 'U_rotated', rank_plot=2,nb_plots=3,plot_index=5,lim=lim)
plot_one_dataset(svd_translated, fig, 'U_traslated', rank_plot=2,nb_plots=3,plot_index=6,lim=lim)

plt.show()

#plot(original_data,rotate(original_data, -math.radians(45)),))
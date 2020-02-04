import numpy as np
from matplotlib import pyplot
import math
import matplotlib.pyplot as plt
from rotations import rotate_via_numpy

def plot(data1):
    x, y = data1[:,0].tolist(),data1[:,1].tolist()

    #print(x)
    fig = plt.figure()

    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1., 0.1))

    #ax = fig.add_subplot(3, 1, 1)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    #ax.spines['left'].set_position('center')
    #ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    #ax.spines['right'].set_color('none')
    #ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    plt.scatter(x, y)
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1., 0.1))
    plt.grid()

    plt.show()


point = [1,0]
point = rotate_via_numpy(point, -math.radians(45))
point = np.matrix([point, [0,0], [1,1]])
print(point)

plot(point)

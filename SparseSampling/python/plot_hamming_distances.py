import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import IPython

hamming_file = '/Users/dpaiton/Google_Drive/School/UCB/Research/Redwood/Sampling/hamming.mat'

hamming = scipy.io.loadmat(hamming_file)

xdat = np.arange(hamming['hammings'].size)
ydat = hamming['hammings'].flatten()
yerr = hamming['hammingsSTD'].flatten()

heatmap=np.arange(2500).reshape((50,50))

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 12}

lines = {'linewidth' : 1,
        'color'      : 'b'}

axes = {'titlesize'   : 'large',
        'titleweight' : 'bold',
        'labelsize'   : 'large',
        'labelweight' : 'bold'}

figure = {'autolayout' : False}

image = {'cmap' : 'hot'}

matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
matplotlib.rc('axes', **axes)
matplotlib.rc('figure', **figure)
matplotlib.rc('image', **image)

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
line1 = plt.errorbar(xdat, ydat, yerr=yerr, fmt='o', markersize=4, color='b', capsize=2)
ax1.set_ylim([0,710])
ax1.set_ylabel('Hamming Distance from Initial Fixed Point')
ax1.set_xlabel('Number of Perturbations')

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(heatmap)
ax2.set_title('Hamming Distance\nBetween Fixed Points')

fig.suptitle('Perturbing Network Leads to Alternate Fixed Points', size='large', weight='bold')

plt.savefig('hamming_distances.eps', format='eps', transparent=True)
plt.show(block=False)

IPython.embed()

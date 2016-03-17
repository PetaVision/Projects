import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import warnings
import IPython

hamming_file = '/Users/dpaiton/Google_Drive/School/UCB/Research/Redwood/Sampling/hamming.mat'
hamming_matrix = '/Users/dpaiton/Google_Drive/School/UCB/Research/Redwood/Sampling/hammingMat.mat'

hamming = scipy.io.loadmat(hamming_file)
hammingMat = scipy.io.loadmat(hamming_matrix)

xdat = np.arange(hamming['hammings'].size)
ydat = hamming['hammings'].flatten()
yerr = hamming['hammingsSTD'].flatten()

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 12}

lines = {'linewidth' : 0.5,
        'color'      : 'b'}

axes = {'titlesize'   : 'medium',
        'titleweight' : 'bold',
        'labelsize'   : 'medium',
        'labelweight' : 'bold'}

figure = {'autolayout' : True}

image = {'cmap' : 'summer'}

matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
matplotlib.rc('axes', **axes)
matplotlib.rc('figure', **figure)
matplotlib.rc('image', **image)

# Almost works, but the errorbar plot was too big
#fig, axes = plt.subplots(nrows=1, ncols=2)
#ham_line = axes[0].errorbar(xdat, ydat, yerr=yerr, fmt='o', markersize=4, color='b', capsize=2)
#axes[0].set_ylim([0,710])
#x0, x1 = axes[0].get_xlim()
#y0, y1 = axes[0].get_ylim()
#axes[0].set_aspect((x1-x0)/(y1-y0))
#axes[0].set_ylabel('Hamming Distance from\nInitial Fixed Point')
#axes[0].set_xlabel('Number of Samples')
#axes[0].autoscale(enable=True, axis='both', tight=False)
#
#im = axes[1].imshow(hammingMat['hamming'], origin='lower')
#axes[1].set_title('Hamming Distance\nBetween Fixed Points')
#axes[1].autoscale_view(tight=True, scalex=True, scaley=True)
#
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#divider = make_axes_locatable(axes[1])
#cax = divider.append_axes("right", size="5%", pad=0.1)
#cbar = plt.colorbar(im, cax=cax)
#
#fig.suptitle('Perturbing Network Leads to Alternate Fixed Points', size='large', weight='bold')

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12, 12), dpi=150)
gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,0.07], height_ratios=[1,1,2])
gs.update(top=0.91, left=0.28, right=0.77)

ax0 = plt.subplot(gs[0, 0])
ham_line = ax0.errorbar(xdat, ydat, yerr=yerr, fmt='o', markersize=1.5, color='b', capsize=1)
x0, x1 = ax0.get_xlim()
y0, y1 = ax0.get_ylim()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UnicodeWarning)
    # ignoring: UnicodeWarning: Unicode equal comparison failed to convert
    # both arguments to Unicode - interpreting them as being unequal
    #
    # Warning occurs because set_aspect is being passed a number instead
    # of 'normal', 'equal', or 'auto' strings
    ax0.set_aspect((x1-x0)/(y1-y0))
ax0.set_ylabel('Hamming Distance from\nInitial Fixed Point')
ax0.set_xlabel('Sample Number')

ax1 = plt.subplot(gs[0, 1])
#modHammingMat = np.ma.masked_where(hammingMat['hamming'] == 0, hammingMat['hamming'])
#cmap = plt.cm.OrRd
#cmap.set_bad(color='black')
im = ax1.imshow(hammingMat['hamming'], origin='lower', interpolation='none', extent=[0, 50, 0, 50])
ax1.set_title('Hamming Distance\nBetween Fixed Points')

ax2 = plt.subplot(gs[0, 2])
cbar = plt.colorbar(im, cax=ax2)

fig.suptitle('Perturbing Network Leads to Alternate Fixed Points', size='large', weight='bold')

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    # ignoring: UserWarning: This figure includes Axes that are not compatible with tight_layout,
    # so its results might be incorrect.
    plt.savefig('hamming_distances.eps', format='eps', transparent=True)

#plt.show(block=False)
#IPython.embed()

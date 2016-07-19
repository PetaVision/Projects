import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import IPython

sparsity_file = '/Users/dpaiton/Google_Drive/School/UCB/Research/Redwood/Sampling/l0_l1_sparsity.mat'
mean_sparsity_file = '/Users/dpaiton/Google_Drive/School/UCB/Research/Redwood/Sampling/mean_l0_l1_ind.mat'

sparsity = scipy.io.loadmat(sparsity_file)
mean_sparsity = scipy.io.loadmat(mean_sparsity_file)

l0dat = sparsity['l0'].flatten()
l1dat = sparsity['l1'].flatten()
mean_l0dat= mean_sparsity['meanl0'].flatten()
mean_l1dat= mean_sparsity['meanl1'].flatten()

xdat = np.arange(l0dat.size)

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 12}

lines = {'linewidth' : 3}

axes = {'titlesize'   : 'large',
        'titleweight' : 'bold',
        'labelsize'   : 'large',
        'labelweight' : 'bold'}

figure = {'autolayout' : True}


matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
matplotlib.rc('axes', **axes)
matplotlib.rc('figure', **figure)
matplotlib.rcParams.update({'figure.autolayout': True})

fig = plt.figure()
ax0 = fig.add_subplot(1,1,1)
#ax1 = ax0.twinx()
#ax2 = ax1.twinx()
#ax3 = ax2.twinx()

labels = ['$l_0$ norm of average', '$l_1$ norm of average', '$l_0$ norm for each perterbation', '$l_1$ norm for each perturbation']

l0 = ax0.plot(xdat, l0dat, color='b', linestyle='-', label=labels[0])
l1 = ax0.plot(xdat, l1dat, color='g', linestyle='-', label=labels[1])
mean_l0 = ax0.plot(xdat, mean_l0dat, color='c', linestyle='-', label=labels[2])
mean_l1 = ax0.plot(xdat, mean_l1dat, color='r', linestyle='-', label=labels[3])

ax0.set_ylim([0, 2500])
#ax1.set_ylim([0, 2500])
#ax2.set_ylim([0, 2500])
#ax3.set_ylim([0, 2500])

ax0.set_xlabel('Number of Samples')
ax0.set_ylabel('Norm Value')

#lines = l0+l1+mean_l0+mean_l1
#labs = [l.get_label() for l in lines]
#ax0.legend(lines, labs, loc=2)

ax0.legend(labels, loc=2)

ax0.set_title('Sparsity Norms increase with Average of\nSamples but Decrease with Each Additional Sample')
#ax0.set_title('Norm of Activity Increases with\nthe Number of Samples in the Average')

plt.savefig('l0_l1_sparsity.eps', format='eps', transparent=True)

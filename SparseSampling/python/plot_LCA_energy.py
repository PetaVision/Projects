import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import IPython

energy_file = '/Users/dpaiton/Google_Drive/School/UCB/Research/Redwood/Sampling/energySingleDiffPerturbations.mat'
energy = scipy.io.loadmat(energy_file)
# energy 1 is 
# energy 2 is half threshold (+/-0.0125)
# energy 3 is threshold (+/-0.025)
# energy 4 is double threshold (+/-0.05)

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 12}

lines = {'linewidth' : 3,
        'color'      : 'b'}

axes = {'titlesize'   : 'large',
        'titleweight' : 'bold',
        'labelsize'   : 'large',
        'labelweight' : 'bold'}

matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
matplotlib.rc('axes', **axes)
matplotlib.rcParams.update({'figure.autolayout': True})

labels = ['$\pm$ Half Threshold', '$\pm$ Threshold', '$\pm$ Double Threshold']

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
line1 = plt.plot(energy['energy2'], color='r', label=labels[0])
line2 = plt.plot(energy['energy3'], color='b', label=labels[1])
line3 = plt.plot(energy['energy4'], color='g', label=labels[2])
ax1.set_title('Energy Increase and Time to Convergence\nare Proportional to the Amount of Noise Injected')
ax1.set_ylabel('Sparse Coding Energy')
ax1.set_xlabel('LCA Time Step')
ax1.legend(labels)

plt.savefig('lca_energy.eps', format='eps', transparent=True)

#plt.show(block=False)
#
#IPython.embed()

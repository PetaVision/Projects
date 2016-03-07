import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython

accuracy_file = '/Users/dpaiton/Google_Drive/School/UCB/Research/Redwood/Sampling/accuracyNoAverage.npy'
accuracy = np.load(accuracy_file)

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

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
line1 = plt.plot(accuracy)
ax1.set_ylim([0.42,0.45])
ax1.set_title('MLP Performance Increases by\nSelecting a Sample With Highest Confidence')
ax1.set_ylabel('Test Accuracy')
ax1.set_xlabel('Number of Samples Available')

plt.savefig('mlp_test_acc_no_avg.eps', format='eps', transparent=True)

#plt.show(block=False)
#
#IPython.embed()

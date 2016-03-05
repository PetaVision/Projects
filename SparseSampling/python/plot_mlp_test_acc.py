import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython

model_version = 1
data_version = 1
log_files = ['net_output/test_s'+str(x)+'_mv'+str(model_version)+'_dv'+str(data_version)+'.log' for x in range(1,50)]

test_accuracy_vals = []
for log_file in log_files:
    with open(log_file, 'r') as f:
        log_text = f.read()
        num_iter = float(re.findall("Running for (\d+) iterations", log_text)[0])
        test_accuracy_vals.append(np.array([float(val) for val in re.findall('test_accuracy \= (\d+\.?\d*)', log_text)]))

#print [x.mean() for x in test_accuracy_vals]

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
line1 = plt.plot([x.mean() for x in test_accuracy_vals])
ax1.set_ylim([0.4,0.5])
ax1.set_title('MLP Performance Increases\nWith Number of Samples in Average')
ax1.set_ylabel('Test Accuracy')
ax1.set_xlabel('Number of Samples in Average')

plt.savefig('mlp_test_acc.eps', format='eps', transparent=True)

#plt.show(block=False)
#
#IPython.embed()

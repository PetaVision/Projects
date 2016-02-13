import re
import numpy as np
import matplotlib.pyplot as plt
import IPython

log_file = 'run_output.log'

log_text = open(log_file, 'r').read()

max_iter = float(re.findall("max_iter: (\d+)",log_text)[0])

train_start = 0
train_step = float(re.findall("display: (\d+)", log_text)[0])
train_times = np.arange(train_start, max_iter, train_step)

train_accuracy_vals = np.array([float(val) for val in re.findall('Train net output \#\d\: accuracy \= (\d+\.?\d*)',log_text)])

fig = plt.figure()
ax1 = fig.add_subplot(111)
line1 = ax1.plot(train_times[0:len(train_accuracy_vals)], train_accuracy_vals, 'r', label='accuracy')
ax1.set_ylabel('Train Accuracy', color='r')
ax1.set_xlabel('Model Time Step')
ax1.set_ylim([0, 1])
plt.show()

IPython.embed()

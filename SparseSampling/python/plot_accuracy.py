import re
import numpy as np
import matplotlib.pyplot as plt
import IPython

log_file = 'sparse_model_output.log'

with open(log_file, 'r') as f:
    log_text = f.read()
    max_iter = float(re.findall("max_iter: (\d+)",log_text)[0])

    train_start = 0
    train_step = float(re.findall("display: (\d+)", log_text)[0])
    train_times = np.arange(train_start, max_iter, train_step)

    train_accuracy_vals = np.array([float(val) for val in re.findall('train_accuracy \= (\d+\.?\d*)', log_text)])
    test_accuracy_vals = np.array([float(val) for val in re.findall('test_accuracy \= (\d+\.?\d*)', log_text)])

fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
line1 = ax1.plot(train_times[0:len(train_accuracy_vals)], train_accuracy_vals, 'r', label='accuracy')
ax1.set_ylabel('Train Accuracy')
ax1.set_ylim([0, 1])

ax2 = fig.add_subplot(2,1,2)
line2 = ax2.plot(train_times[0:len(test_accuracy_vals)], test_accuracy_vals, 'r', label='accuracy')
ax2.set_ylabel('Validation Accuracy')
ax2.set_xlabel('Model Time Step')
ax2.set_ylim([0, 1])

plt.show(block=False)

IPython.embed()

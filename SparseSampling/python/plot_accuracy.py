import re
import numpy as np
import matplotlib.pyplot as plt
import IPython

log_file = 'sparse_model_output_v0.log'

with open(log_file, 'r') as f:
    log_text = f.read()
    max_iter = float(re.findall("max_iter: (\d+)", log_text)[0])

    time_start = 0
    time_step = float(re.findall("display: (\d+)", log_text)[0])
    time_list = np.arange(time_start, max_iter, time_step)

    train_accuracy_vals = np.array([float(val) for val in re.findall('train_accuracy \= (\d+\.?\d*)', log_text)])
    test_accuracy_vals = np.array([float(val) for val in re.findall('test_accuracy \= (\d+\.?\d*)', log_text)])

#Convert time step to epoch
batch_size = 100
num_images_per_epoch = 50000 # CIFAR has 50k training images
epoch_list = np.multiply(time_list, np.float(batch_size)/num_images_per_epoch)

fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
line1 = ax1.plot(epoch_list[0:len(train_accuracy_vals)], train_accuracy_vals[0:len(epoch_list)], 'r', label='train_accuracy')
#line1 = ax1.plot(time_list[0:len(train_accuracy_vals)], train_accuracy_vals[0:len(time_list)], 'r', label='train_accuracy')
ax1.set_ylabel('Train Accuracy')
ax1.set_ylim([0, 1])

ax2 = fig.add_subplot(2,1,2)
line2 = ax2.plot(epoch_list[0:len(test_accuracy_vals)], test_accuracy_vals[0:len(epoch_list)], 'b', label='validation_accuracy')
#line2 = ax2.plot(time_list[0:len(test_accuracy_vals)], test_accuracy_vals[0:len(time_list)], 'b', label='validation_accuracy')
#ax2.set_ylabel('Validation Accuracy')
ax2.set_xlabel('Number of epochs')
#ax2.set_xlabel('Number of iterations')
ax2.set_ylim([0, 1])

plt.show(block=False)

IPython.embed()

import re
import numpy as np
import IPython

model_version = 0
log_file = 'net_output/test_pixel_output_v'+str(model_version)+'.log'

with open(log_file, 'r') as f:
    log_text = f.read()
    num_iter = float(re.findall("Running for (\d+) iterations", log_text)[0])
    test_accuracy_vals = np.array([float(val) for val in re.findall('test_accuracy \= (\d+\.?\d*)', log_text)])

print np.mean(test_accuracy_vals)

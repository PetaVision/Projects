import re
import numpy as np
import matplotlib.pyplot as plt
import IPython

model_version = 1
data_version = 1
log_files = ['test_s'+str(x)+'_output_mv'+str(model_version)+'_dv'+str(data_version)+'.log' for x in range(1,20)]

test_accuracy_vals = []
for log_file in log_files:
    with open(log_file, 'r') as f:
        log_text = f.read()
        num_iter = float(re.findall("Running for (\d+) iterations", log_text)[0])
        test_accuracy_vals.append(np.array([float(val) for val in re.findall('test_accuracy \= (\d+\.?\d*)', log_text)]).mean())
IPython.embed()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.environ['HOME']+'/Work/Libraries/caffe/python/')) # Caffe
import caffe
import IPython

solver_file_name = 'solver.prototxt'

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

solver = caffe.SGDSolver(solver_file_name)
net = solver.net
solver.solve()

IPython.embed()

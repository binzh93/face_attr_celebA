import sys
import os

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('model/solver.prototxt')
# solver.net.copy_from('model/sphereface_model.caffemodel')
solver.solve()
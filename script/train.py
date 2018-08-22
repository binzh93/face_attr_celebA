import sys
import os
# curr_path = os.path.abspath(".")
# sys.path.append(curr_path)

import caffe
caffe.set_device(1)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('model/solver.prototxt')
solver.net.copy_from('model/sphereface_model.caffemodel')
solver.solve()
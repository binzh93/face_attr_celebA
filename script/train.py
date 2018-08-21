import sys
import os
curr_path = os.path.abspath(".")
sys.path.append(curr_path)

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('model/solver.prototxt')
#solver.net.copy_from('/workspace/mnt/group/face-det/zhubin/train_file/initial.caffemodel')
solver.solve()
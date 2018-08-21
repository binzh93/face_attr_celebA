import sys
import os
# curr_path = os.path.abspath(".")
# sys.path.append(curr_path)

import caffe
caffe.set_device(1)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('/workspace/mnt/group/face-reg/zhubin/face_attr_celebA/model/solver.prototxt')
solver.net.copy_from('/workspace/mnt/group/face-reg/zhubin/face_attr_celebA/model/sphereface_model.caffemodel')
solver.solve()
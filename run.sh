#!/bin/sh
cd /workspace/mnt/group/face-reg/zhubin/face_attr_celebA/
# if you run in ava training job, turn on the next three line 
apt-get update
yes| apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
apt-get install --no-install-recommends libboost-all-dev
export PYTHONPATH=/workspace/mnt/group/face-reg/zhubin/caffe/python:$PYTHONPATH

echo '===>Start training!' 
python script/train.py
# nohup python script/train.py > FaceAttr_new1.log 2>&1 &
echo '===>Training finished!' 

# python script/train.py
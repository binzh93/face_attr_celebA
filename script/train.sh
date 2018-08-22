#!/bin/sh
#cd /workspace/mnt/group/face-reg/zhubin/face_attr_celebA/
echo '===>Start training!' >> FaceAttr.log
#python train.py
nohup python script/train.py > FaceAttr.log 2>&1 &
echo '===>Training finished!' >> FaceAttr.log

# python script/train.py
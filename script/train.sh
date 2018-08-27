#!/bin/sh
#cd /workspace/mnt/group/face-reg/zhubin/face_attr_celebA/
echo '===>Start training!' >> FaceAttr_new1.log
#python train.py
nohup python script/train.py > FaceAttr_new1.log 2>&1 &
echo '===>Training finished!' >> FaceAttr_new1.log

# python script/train.py
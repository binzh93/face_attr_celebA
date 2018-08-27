import sys
import cv2
import re

sys.path.append('/workspace/mnt/group/face-reg/zhubin/caffe/python')
import caffe
import numpy as np
import random
import cPickle as pickle
import os
import time



def mirror(img):
    img = cv2.flip(img, 1)
    return img


def illumination(img):
    r0 = random.uniform(0.0, 1.0)
    r1 = random.uniform(0.0, 1.0)
    hue = r0 * 0.1 + 1
    exposure = r1 * 0.5 + 1
    if random.uniform(0.0, 1.0) > 0.5:
        exposure = 1.0 / exposure
    if random.uniform(0.0, 1.0) > 0.5:
        hue = 1.0 / hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] * exposure
    v = np.clip(v, 0, 255)
    # v = v.astype(np.int32)
    h = hsv[:, :, 0] * hue
    h = np.clip(h, 0, 255)
    # h = h.astype(np.int32)

    hsv[:, :, 2] = v.astype(np.uint8)
    hsv[:, :, 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    

# read img list
def readSrcFile(basePath, src_file):
    f = open(src_file, 'r')
    imgLabelList = []

    print("-------------{}---------------".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
#     print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for line in f.readlines():
        temp = line.split(" ")
        labelList = [int(i) for i in temp[1:]]
        image = cv2.imread(os.path.join(basePath, temp[0]))
        imgLabelList.append([image, labelList])
    print("-------------{}---------------".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
#     print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return imgLabelList

################################################################################
#########################Train Data Layer By Python#############################
################################################################################
class Data_Layer_train(caffe.Layer):

    def setup(self, bottom, top):
        if len(top) != 41:
            raise Exception("Need to define tops (data, label)")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom")

        self.mean = 127.5
        self.scale = 1 / 128.0

        params = eval(self.param_str)
        self.mirror = params["mirror"]
        self.illumination = params["illumination"]
        self.batch_size = params["batch_size"]
        self.src_file = params['src_file']
        self.basepath = params['img_basepath']

        self.im_size = params["im_size"]    # "(200, 200)" ==> [200, 200]  str ==>list

        self.imgLabelList = readSrcFile(self.basepath, self.src_file)
        self._cur = 0  # use this to check if we need to restart the list of images

        self.data_aug_type = ["normal"]
        if self.mirror == True:
            self.data_aug_type.append("mirror")
        if self.illumination == True:
            self.data_aug_type.append("illumination")
        if ("mirror" in self.data_aug_type) and ("illumination" in self.data_aug_type):
            self.data_aug_type.append("mirror_illumination")
        
        top[0].reshape(self.batch_size, 3, self.im_size[0], self.im_size[1])
        for i in xrange(1, 41):
            top[i].reshape(self.batch_size, 1)


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, labelList = self.load_next_image()
            top[0].data[itt, ...] = im
            for nums in xrange(40):
                top[nums+1].data[itt, ...] = labelList[nums]


    def backward(self, top, propagate_down, bottom):
        pass


    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        if self._cur == len(self.imgLabelList):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgLabelList)
        img, labelList = self.imgLabelList[self._cur]
        self._cur += 1

        # h = image.shape[0]
        # w = image.shape[1]
        # if h != w:
        #     raise Exception("image height not equal width")
        # if h != self.im_size:
        #     raise Exception("image height not equal the prototxt input size")
        image = self.data_augment(img)

        # normalization
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image -= self.mean
        image *= self.scale

        return image, labelList

    # mirror, illumination, mirror+illumination
    def data_augment(self, image):
        # choose a type of data augment
        idx = random.randint(0, len(self.data_aug_type) - 1)

        if self.data_aug_type[idx] == 'mirror':
            image = mirror(image)
        elif self.data_aug_type[idx] == 'illumination':
            image = illumination(image)
        elif self.data_aug_type[idx] == 'mirror_illumination':
            image = illumination(image)
            image = mirror(image)
        else:
            image = image
        return image

    

################################################################################
#########################Validation Data Layer By Python########################
################################################################################

class Data_Layer_validation(caffe.Layer):

    def setup(self, bottom, top):
        if len(top) != 41:
            raise Exception("Need to define tops (data, label)")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom")

        self.mean = 127.5
        self.scale = 1 / 128.0

        params = eval(self.param_str)
        self.mirror = params["mirror"]
        self.illumination = params["illumination"]
        self.batch_size = params["batch_size"]
        self.src_file = params['src_file']
        self.basepath = params['img_basepath']

        self.im_size = params["im_size"]    # "(200, 200)" ==> [200, 200]  str ==>list

        self.imgLabelList = readSrcFile(self.basepath, self.src_file)

        self._cur = 0  # use this to check if we need to restart the list of images

        self.data_aug_type = ["normal"]
        if self.mirror == True:
            self.data_aug_type.append("mirror")
        if self.illumination == True:
            self.data_aug_type.append("illumination")
        if ("mirror" in self.data_aug_type) and ("illumination" in self.data_aug_type):
            self.data_aug_type.append("mirror_illumination")
        
        top[0].reshape(self.batch_size, 3, self.im_size[0], self.im_size[1])
        for i in xrange(1, 41):
            top[i].reshape(self.batch_size, 1)


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, labelList = self.load_next_image()
            top[0].data[itt, ...] = im
            for nums in xrange(40):
                top[nums+1].data[itt, ...] = labelList[nums]


    def backward(self, top, propagate_down, bottom):
        pass


    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        if self._cur == len(self.imgLabelList):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgLabelList)
        img, labelList = self.imgLabelList[self._cur]
        self._cur += 1

        # h = image.shape[0]
        # w = image.shape[1]
        # if h != w:
        #     raise Exception("image height not equal width")
        # if h != self.im_size:
        #     raise Exception("image height not equal the prototxt input size")
        image = self.data_augment(img)

        # normalization
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image -= self.mean
        image *= self.scale

        return image, labelList

    # mirror, illumination, mirror+illumination
    def data_augment(self, image):
        # choose a type of data augment
        idx = random.randint(0, len(self.data_aug_type) - 1)

        if self.data_aug_type[idx] == 'mirror':
            image = mirror(image)
        elif self.data_aug_type[idx] == 'illumination':
            image = illumination(image)
        elif self.data_aug_type[idx] == 'mirror_illumination':
            image = illumination(image)
            image = mirror(image)
        else:
            image = image
        return image





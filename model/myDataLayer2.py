import sys
import cv2
import re

sys.path.append('/workspace/mnt/group/face-reg/zhubin/caffe/python')
import caffe
import numpy as np
import random
import cPickle as pickle
import os


# def random_crop(img, size, cop_size, flag, xy=[]):
#     if flag == 0:
#         h_off = random.randint(0, size - cop_size)
#         w_off = random.randint(0, size - cop_size)
#         xy = [xy[0] - w_off, xy[1] - h_off, xy[2] - w_off, xy[3] - h_off, xy[4] - w_off, xy[5] - h_off, xy[6] - w_off,
#               xy[7] - h_off, xy[8] - w_off, xy[9] - h_off]
#         crop_img = img[h_off:h_off + cop_size, w_off:w_off + cop_size]
#         return crop_img, xy
#     if flag == 1:
#         h_off = random.randint(0, size - cop_size)
#         w_off = random.randint(0, size - cop_size)
#         crop_img = img[h_off:h_off + cop_size, w_off:w_off + cop_size]
#         return crop_img


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
def readSrcFile(src_file):
        f = open(src_file, 'r')
        imgPathLabelList = []
        for line in f.readlines():
            temp = line.split(" ")
            labelList = [int(i) for i in temp[1:]]
            imgPathLabelList.append([temp[0], labelList])
        return imgPathLabelList

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
        self.im_size =  re.split(r"\(|,|\)", self.im_size)
        self.im_size = [int(i.strip()) for i in self.im_size if i != ""]
        # self.crop_size = params["crop_size"]

        self.imgLabelList = readSrcFile(self.src_file)
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
        # top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)
        # top[1].reshape(self.batch_size, 1)
        # top[2].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, labelList = self.load_next_image()
            top[0].data[itt, ...] = im
            for nums in xrange(40):
                top[nums+1].data[itt, ...] = labelList[nums]

            # top[0].data[itt, ...] = im
            # top[1].data[itt, ...] = label
            # top[2].data[itt, ...] = pts

    def backward(self, top, propagate_down, bottom):
        pass

    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        if self._cur == len(self.imgLabelList):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgLabelList)
        img_path, labelList = self.imgLabelList[self._cur]
        self._cur += 1
        # bgr
        image = cv2.imread(os.path.join(self.basepath, img_path))
        # h = image.shape[0]
        # w = image.shape[1]
        # if h != w:
        #     raise Exception("image height not equal width")
        # if h != self.im_size:
        #     raise Exception("image height not equal the prototxt input size")
        image = self.data_augment(image)

        # normalization
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image -= self.mean
        image *= self.scale
        #print os.path.join(self.basepath, img_path), label, pts
        
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
        self.im_size =  re.split(r"\(|,|\)", self.im_size)
        self.im_size = [int(i.strip()) for i in self.im_size if i != ""]
        # self.crop_size = params["crop_size"]

        self.imgLabelList = readSrcFile(self.src_file)
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
        # top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)
        # top[1].reshape(self.batch_size, 1)
        # top[2].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, labelList = self.load_next_image()
            top[0].data[itt, ...] = im
            for nums in xrange(40):
                top[nums+1].data[itt, ...] = labelList[nums]

            # top[0].data[itt, ...] = im
            # top[1].data[itt, ...] = label
            # top[2].data[itt, ...] = pts

    def backward(self, top, propagate_down, bottom):
        pass

    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        if self._cur == len(self.imgLabelList):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgLabelList)
        img_path, labelList = self.imgLabelList[self._cur]
        self._cur += 1
        # bgr
        image = cv2.imread(os.path.join(self.basepath, img_path))
        # h = image.shape[0]
        # w = image.shape[1]
        # if h != w:
        #     raise Exception("image height not equal width")
        # if h != self.im_size:
        #     raise Exception("image height not equal the prototxt input size")
        image = self.data_augment(image)

        # normalization
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image -= self.mean
        image *= self.scale
        #print os.path.join(self.basepath, img_path), label, pts
        
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



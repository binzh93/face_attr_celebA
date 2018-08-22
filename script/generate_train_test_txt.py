#-*- coding: utf-8 -*-
import os
import random
import cv2

def change_label_type(x):
    if x == "-1":
        return 0
    elif x == "1":
        return 1


def generate_train_test_txt(doubleFaceDir, list_attr_celeba_file, saveTrainLsitPath, saveTestListPath):
    doubleFaceList = []
    for name in os.listdir(doubleFaceDir):
        if name.endswith(".jpg"):
            doubleFaceList.append(name)
    all_label_dict = {}
    with open(list_attr_celeba_file, "r") as fr:
        fr.readline()
        fr.readline()
        for line in fr:
            temp = line.strip().split(" ")
            label = [i for i in temp[1: ] if i=="1" or i == "-1"]
            label = map(change_label_type, label)
            all_label_dict[temp[0]] = label
    random.Random(100).shuffle(doubleFaceList)
    num_len = len(doubleFaceList)
    train_list = doubleFaceList[0: int(num_len*0.8)]
    test_list = doubleFaceList[int(num_len*0.8): ]
    train_str = ""
    for val in train_list:
        train_str += doubleFaceDir + val
        for label_val in all_label_dict[val]:
            train_str += " " + str(label_val)
        train_str += "\n"
    with open(saveTrainLsitPath, "w") as fw1:
        fw1.write(train_str)
    test_str = ""
    for val in test_list:
        test_str += doubleFaceDir + val
        for label_val in all_label_dict[val]:
            test_str += " " + str(label_val)
        test_str += "\n"
    with open(saveTestListPath, "w") as fw2:
        fw2.write(test_str)
    


def test_ave_width_height(train_txt, test_txt, basePath):
    w_a, h_a = 0, 0
    train_nums = 0
    with open(test_txt, "r") as fr:
        for line in fr:
            imgPath = os.path.join(basePath, line.split(" ")[0])
#             print(imgPath)
            img = cv2.imread(imgPath)
#             print img.shape
            w_a += img.shape[0]
            h_a += img.shape[1]
            train_nums += 1
    print("ave width: {}, ave height: {}".format(w_a*1.0/train_nums, h_a*1.0/train_nums))


def generate_train_valition_test_txt(doubleFaceDir, list_attr_celeba_file, saveTrainLsitPath, saveValLsitPath, saveTestListPath):

    all_img_label_list = []
    with open(list_attr_celeba_file, "r") as fr:
        fr.readline()
        fr.readline()
        for line in fr:
            temp = line.strip().split(" ")
            label = [i for i in temp[1: ] if i=="1" or i == "-1"]
            label = map(change_label_type, label)
            all_img_label_list.append([temp[0], label])
    
    trainList = all_img_label_list[0: 162770]
    validationList = all_img_label_list[162770: 162770+19867]
    testList = all_img_label_list[162770+19867: ]
    train_str = ""
    for val in trainList:
        train_str += doubleFaceDir + val[0]
        for label_val in val[1]:
            train_str += " " + str(label_val)
        train_str += "\n"
    with open(saveTrainLsitPath, "w") as fw1:
        fw1.write(train_str)
    
    validation_str = ""
    for val in validationList:
        validation_str += doubleFaceDir + val[0]
        for label_val in val[1]:
            validation_str += " " + str(label_val)
        validation_str += "\n"
    with open(saveValLsitPath, "w") as fw2:
        fw2.write(validation_str)

    test_str = ""
    for val in testList:
        test_str += doubleFaceDir + val[0]
        for label_val in val[1]:
            test_str += " " + str(label_val)
        test_str += "\n"
    with open(saveTestListPath, "w") as fw3:
        fw3.write(test_str)
    


        
if __name__ == "__main__":
    doubleFaceDir = "celebA/aligned-img-200x150/aligned_imgs/"
    list_attr_celeba_file = "celebA/list_attr_celeba.txt"
    saveTrainLsitPath = "celebA/train.txt"
    saveValLsitPath = "celebA/validation.txt"
    saveTestListPath = "celebA/test.txt"
    generate_train_valition_test_txt(doubleFaceDir, list_attr_celeba_file, saveTrainLsitPath, saveValLsitPath, saveTestListPath)
#     generate_train_test_txt(doubleFaceDir, list_attr_celeba_file, saveTrainLsitPath, saveTestListPath)
    
#     basePath = "/workspace/mnt/group/face-reg/zhubin"
#     test_ave_width_height(saveTrainLsitPath, saveTestListPath, basePath)
    










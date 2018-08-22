from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw_distribution_pic(train_fn, valdation_fn, test_fn):
    train=open(train_fn).readlines()
    test = open(test_fn).readlines()
    result1={}
    for line in test:
        spl=line.strip().split()
        label = spl[1:]
        for ii in range(40):
            if label[ii]=='1':
                if ii not in result1:
                    result1[ii]=1
                else:
                    result1[ii]+=1
    result2={}
    for line in train:
        spl=line.strip().split()
        label = spl[1:]
        for ii in range(40):
            if label[ii]=='1':
                if ii not in result2:
                    result2[ii]=1
                else:
                    result2[ii]+=1

    comp ={}
    for i in range(40):
        ratio=result2[i]/(result2[i]+result1[i])
        comp[i]=ratio
    print comp

    x=range(40)
    y1=result1.values()
    y2=result2.values()
    y3=comp.values()
    plt.plot(x,y1, label="test data")
    plt.plot(x,y2, label="train data")
    plt.plot(x,y3, label="train_ratio")
    plt.legend(loc='best')
    plt.savefig('result.jpg')

if __name__ == "__main__":
    train_fn = "test_code/train.txt"
    valdation_fn = "test_code/valition.txt"
    test_fn = "test_code/test.txt"

    # draw_distribution_pic(train_fn, valdation_fn, test_fn)
    
''' 把xml文件每个图片的标号(eg:000001.xml--->000001)提取出来，放在训练集、测试集、验证集文件(.txt)
1 xml文件地址和结果保存地址
2 设置训练集和测试集数据比例
3 读入xml文件中的名称保存在total_xml，按照训练集和测试集数据比例取得训练集和测试集数据下标
4 打开训练集、测试集、验证集文件
5 遍历所有的样本，根据下标所属的数据划分类别，放在对应的文件夹, 如果下标在训练集，则把name放在训练集，
  反之放在test.txt ; 如果属于测试集且属于train.txt,则name放在train.txt，反之放在val.txt。关闭之前打开的所有文件。

'''
import random
import os,sys

#  1
xmlfilepath=r'./VOCdevkit/VOC2007/Annotations'
saveBasePath=r"./VOCdevkit/VOC2007/ImageSets/Main/"
 
trainval_percent=1
train_percent=1

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num=len(total_xml)  
list=range(num)
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size : ",tv)
print("traub suze : ",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()

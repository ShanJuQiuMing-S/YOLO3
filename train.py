import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.yolo3 import yolo_body
from nets.loss import yolo_loss
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_random_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):     # classes_path = 'model_data/voc_classes.txt'
    '''loads the classes'''        # ['aeroplane\n', 'bicycle\n', 'bird\n', 'boat\n', 'bottle\n', 'bus\n', 'car\n', 'cat\n', 'chair\n', 'cow\n', 'diningtable\n', 'dog\n', 'horse\n', 'motorbike\n', 'person\n', 'pottedplant\n', 'sheep\n', 'sofa\n', 'train\n', 'tvmonitor']
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):      # name='model_data/yolo_anchors.txt' mode='r' encoding='UTF-8'>
    '''loads the anchors from a file'''
    with open(anchors_path) as f:   # anchors_path = 'model_data/yolo_anchors.txt'
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]  # [1anchors = {lis0.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
    return np.array(anchors).reshape(-1, 2)   # anchors.shape=[9,2];[[ 10.  13.], [ 16.  30.], [ 33.  23.], [ 30.  61.], [ 62.  45.], [ 59. 119.], [116.  90.], [156. 198.], [373. 326.]]


#---------------------------------------------------#
#   训练数据生成器
#---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator
    anchors = {ndarray: (9, 2)} 已经设置好的尺寸
    annotation_lines = {list: 3} xml文件的个数，每一行有真实框的坐标和对应物体的类别
    batch_size = {int} 1
    input_shape = {tuple: 2} (416, 416)
    num_classes = {int} 20
    把图片和真实框按批转化为数组，根据真实框和锚框得到标签y_true
    '''
    n = len(annotation_lines)  # n = 3
    i = 0                      # 计数器
    while True:
        # 1 每batch_size的图片和真实框标签
        image_data = [] # 储存图片的列表 [416,416,3]
        box_data = []   # 储存框的列表[1,num_gt,5]
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)   #
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)  # box.shape = (20,5)
            image_data.append(image)      # imge.shape =【416，416，3】
            box_data.append(box)
            i = (i+1) % n
        # 2 图片和真实框的数据转化为数值
        image_data = np.array(image_data) # image_data.shape=[1,416,416,3] 把列表转化为数组，维度会扩展一维
        box_data = np.array(box_data)  # box.shape=[20,5] (box_data).shape= [1,20,5]
        # 3 根据真实框和锚框的宽高获得标签y_true
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes) # box_data(1,20,5),input_shape(416,416), anchors(9,2),num_classes=20
        yield [image_data, *y_true], np.zeros(batch_size) # y_true.list=3 [m,h,w,k,25]


#---------------------------------------------------#
#   读入xml文件，并输出y_true
#---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''  找到真实框在特征图上的位置，制作标签y_true '''
    # true_boxes[1,20,5], input_shape[416,416], anchors[9,2], num_classes[20]
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors)//3  # num_layers=9//3=3
    # 先验框
    # 678为116,90,  156,198,  373,326
    # 345为30,61,  62,45,  59,119
    # 012为10,13,  16,30,  33,23,  
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')  # （x1,y1,x2,y2） shape[1,20,5]
    input_shape = np.array(input_shape, dtype='int32')  #  416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)   true_boxes=[x1,x2,x3,x4]
    # 1 根据真实框的左上角和右下角坐标得出框的中心点和宽高,把真实框映射到特征层
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2   # 中心点
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]          # 宽高
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[:]  # 真实框防缩到特征图上
    true_boxes[..., 2:4] = boxes_wh/input_shape[:]

    # 2 设置网格尺寸
    m = true_boxes.shape[0] # m张图
    # 得到网格的shape为13,13;26,26;52,52
    # 每个特征层对应的网格
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]   # input_shape=418 ,grid_shapes=[13, 26, 52]
    # y_true的格式为(m,13,13,3,25)(m,26,26,3,25)(m,52,52,3,25)
    # 3 设置标签shape
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # 4 处理先验框，用于后续计算
    anchors = np.expand_dims(anchors, 0)   # [1,9,2]
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]  # boxes_wh.shape[m,2]
        if len(wh)==0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)  # boxes_wh.shape[1,m,2]
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 5 计算真实框(1个)和哪个先验框（多个）最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 维度是(n)
        best_anchor = np.argmax(iou, axis=-1)  # 每个特征层上跟真实框匹配的先验框

        # 找出真实框在特征图的位置
        for t, n in enumerate(best_anchor):  # 遍历所有的最优先验框
            for l in range(num_layers):      # 遍历特征层
                if n in anchor_mask[l]:      # 查看框是否在特征层上
                    # floor用于向下取整 true_boxes[1,20,5]，真实框的中心点就是标签y在特征图上的坐标
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')   # y_true的格式为(m,13,13,3,25)
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]   # 真实框在特征图上的位置
                    y_true[l][b, j, i, k, 4] = 1     # 置信度
                    y_true[l][b, j, i, k, 5+c] = 1   # 类别

    return y_true


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

if __name__ == "__main__":
    # 标签的位置
    annotation_path = '2007_train.txt'
    # 获取classes和anchor的位置
    classes_path = 'model_data/voc_classes.txt'    
    anchors_path = 'model_data/yolo_anchors.txt'
    # 预训练模型的位置
    weights_path = 'model_data/yolo_weights.h5'
    # 获得classes和anchor
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    # 一共有多少类
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # 训练后的模型保存的位置
    log_dir = 'logs/'
    # 输入的shape大小
    input_shape = (416,416)

    # 清除session
    K.clear_session()

    # 输入的图像为
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape

    # 创建yolo模型
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors//3, num_classes) # model_body = [y1,y2,y3]，得到框、分数、类别概率、修正框、筛选框
    '''
    image_input.shape[?,?,?,3],num_classes=20,num_anchors//3=3
    '''
    # 载入预训练权重
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)  # 加载模型所需的参数
    
    # y_true为13,13,3,25
    # 26,26,3,25
    # 52,52,3,25
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    # 输入为*model_body.input, *y_true
    # 输出为model_loss
    loss_input = [*model_body.output, *y_true]   # loss_input
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)

    freeze_layers = 249
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    # 调整非主干模型first
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 1
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # model.load_weights('logs/') log文件里的值

    # 解冻
    for i in range(freeze_layers): model_body.layers[i].trainable = True
    # 解冻后训练
    if True:
        model.compile(optimizer=Adam(lr=1e-4), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 1
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=100,
                initial_epoch=50,   #  这个值要跟上一步部分结构训练的结果的迭代次数一致。
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'last1.h5')

#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import skimage.io as io

import os

'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(os.path.join(".","mnist_data"), one_hot=True)
batch_xs, batch_ys=mnist.train.next_batch(100)
'''
#圖片標準尺寸

ImgHeight=100
ImgWidth=100

#Input TFRecords file:
InputTFR = os.path.join(".",r"whales.tfrecords")

def labels_onehot_array(labels_bytes_array):
    output_structure=[]
    
    for l in labels_bytes_array:
        output_structure.append([float(i) for i in list(l.decode())])
    
    label_oh_array=np.array(output_structure)

#    print(label_oh_array.shape)

    return label_oh_array
#=============================================================================
def compute_accuracy(test_x, test_y):
    global prediction

    y_pred=sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1 })
    
    correct_prediction=tf.equal(tf.argmax(y_pred,1), tf.argmax(v_ys,1))
    #print(sess.run(tf.argmax(y_pred,1)), sess.run(tf.argmax(v_ys,1)))
    #print(sess.run(correct_prediction))

    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(sess.run(tf.cast(correct_prediction, tf.float32)))
    result=sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result
#=============================================================================
def read_and_decode(filename_queue):

    global ImgHeight, ImgWidth, InputTFR

    reader=tf.TFRecordReader()

    _, serialized_example=reader.read(filename_queue)

    #讀取一筆 Example
    features=tf.parse_single_example(
        serialized_example,
        features={
            
            "height":tf.FixedLenFeature([],tf.int64),
            "width":tf.FixedLenFeature([],tf.int64),
            "image_string":tf.FixedLenFeature([], tf.string),
            "label":tf.FixedLenFeature([],tf.float32),
            "label2":tf.FixedLenFeature([], tf.string),
            
            }
        )
    #將序列化的圖片轉為uint8的tensor
    image=tf.decode_raw(features["image_string"], tf.uint8)

    # bytes -> str -> list -> array
    label_oh_bytes=tf.cast(features['label2'], tf.string)
    

    #將label的資料轉為float32的tensor
    label=tf.cast(features["label"], tf.float32)
    
    #將圖片的大小轉為int32的tensor

    height=tf.cast(features["height"], tf.int32)
    width=tf.cast(features["width"], tf.int32)
    #將圖片調為正確尺寸
    image = tf.reshape(image, [ height, width, 3])
     
    #圖片標準尺寸
    #image_size_const =tf.constant((ImgHeight, ImgWidth, 3), dtype=tf.int32)

    #將圖片調整為標準尺寸
    
    image=tf.image.resize_image_with_crop_or_pad(
        image=image,
        target_height=ImgHeight*7,
        target_width=ImgWidth*7
        )
    
    resized_image = tf.image.resize_images(image, [ImgHeight, ImgWidth], method=1)
    #打散資料順序
    min_after_dequeue=10
    batch_size=2#23
    capacity=min_after_dequeue+3*batch_size
    
    
    images, labels= tf.train.shuffle_batch(
        
        [resized_image, label_oh_bytes],
        #[resized_image, label],

        batch_size=batch_size,
        capacity=capacity,
        num_threads=128,
        min_after_dequeue=min_after_dequeue

        )
        
   
    return images, labels

#============================================================================
#Lenet:https://hk.saowen.com/a/8f6b4330a765203fcfb94b8d791ad0d0b6fc9f5463f7a405ef5c86e024edbdec
#     :https://zhuanlan.zhihu.com/p/31612931
#softmax def:http://www.cnblogs.com/maybe2030/p/5678387.html

xs=tf.placeholder(dtype=tf.float32, shape=[None, ImgHeight,ImgWidth,3])
ys=tf.placeholder(dtype=tf.float32, shape=[None, 21])

keep_prob=tf.placeholder(dtype=tf.float32)

def ConvNN(input_data):#[-1, ImgHeight,ImgWidth,3]

    global ImgHeight, ImgWidth

    #image

    #Convolution
    W1_conv=tf.Variable(tf.truncated_normal([5,5 ,3,32],stddev=0.1))#patch 5x5 in_size 3 outsize 32
    B1_conv=tf.Variable(tf.constant(0.1,shape=[32]))#
    H1_conv=tf.nn.relu(tf.add(tf.nn.conv2d(input_data, W1_conv, strides=[1,1,1,1], padding="SAME"), B1_conv))#700x700x32
    
    #max_pooling
    MXP1=tf.nn.max_pool(H1_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")#350x350x32
    #print(MXP1.shape)

    #Convolution
    W2_conv=tf.Variable(tf.truncated_normal([5,5, 32, 64], stddev=0.1))#patch 5x5 insize 32 outsize 64
    B2_conv=tf.Variable(tf.constant(0.1, shape=[64]))
    H2_conv=tf.nn.relu(tf.add(tf.nn.conv2d(MXP1, W2_conv, strides=[1,1,1,1], padding="SAME"),B2_conv))#350x350x64
    
    #max_pooling
    MXP2=tf.nn.max_pool(H2_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")#175x175x64
    #print(MXP2.shape)

    #resize for fullyconn
    height_fc=int(ImgHeight/4)
    width_fc=int(ImgWidth/4)
    
    MXP2_flat=tf.reshape(MXP2,[-1, height_fc*width_fc*64])
    print(MXP2_flat.shape)
    #Fully Connected
    W1_fc=tf.Variable(tf.truncated_normal([height_fc*width_fc*64, 1024], stddev=0.1))
    B1_fc=tf.Variable(tf.constant(0.1, shape=[1024]))
    H1_fc=tf.nn.relu(tf.add(tf.matmul(MXP2_flat, W1_fc), B1_fc))
    H1_fc_dropout=tf.nn.dropout(H1_fc, keep_prob)
    #Fully Connected
    W2_fc=tf.Variable(tf.truncated_normal([1024,21], stddev=0.1))
    B2_fc=tf.Variable(tf.constant(0.1,shape=[21]))
    prediction=tf.nn.softmax(tf.add(tf.matmul(H1_fc_dropout, W2_fc), B2_fc))
    #Classifier
    
    return prediction


x_image=tf.reshape(xs, [-1, ImgHeight,ImgWidth,3])

prediction=ConvNN(x_image)

loss=tf.reduce_mean( -tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]) )

train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)



#============================================================================
#建立檔名佇列
filename_tfr=r"whales.tfrecords"
print(sum(1 for _ in tf.python_io.tf_record_iterator(filename_tfr)))
filename_queue = tf.train.string_input_producer(

    [filename_tfr], num_epochs=1
    
    )
#讀取並解析TFRecords的資料
images, labels=read_and_decode(filename_queue)

#初始化變數
init_op=tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
                 )

with tf.Session() as sess:
    #初始化
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    #
    for i in range(11):
        img, lab = sess.run([images, labels])
        
        #檢查每個batch的圖片維度
        print("No.{n:.2f}: shape is {shape} with label:{label} & with type: {tp}".format(n=i+1, shape=img.shape, label=labels_onehot_array(lab), tp=type(lab)))
        
        #顯示每個Batch的第一張圖
        #io.imshow(img[0,:,:,:])
        #plt.show()

        sess.run(train_step, feed_dict={xs:img, ys:labels_onehot_array(lab), keep_prob:0.5})
        #print(str(i+1)+"-th")


    coord.request_stop()
    coord.join(threads)
#>>> mnist.test.labels[0].dtype
#dtype('float64')



import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Conv2DTranspose,Lambda
from keras.models import Model
import keras.backend as K
import pickle
from sklearn.model_selection import train_test_split
import matplotlib as plt
from keras.optimizers import SGD
from MEF_SSIM import MEF_SSIM
import os
from itertools import product
from skimage import io,img_as_float32
import json
import glob
import keras.backend as K
from keras.utils import plot_model
import pickle
def change_dim(input):
    return tf.stack(tf.split(input,3,axis=-1))
def cat(input):
    return tf.concat((input[0],input[1]),axis=-1)
def SSIM(input):
    return tf.image.ssim(input[0],input[1],max_val=1)
def reshape_half(input):
    return tf.image.resize_images(input,tf.constant([400,400]),method=0)
def reshape_qtr(input):
    return tf.image.resize_images(input,tf.constant([200,200]),method=0)
def vgg_16(height=None,width=None):

    ldr_tensor=Input(shape=(height,width,9),name="hdr")
    gd=Input(shape=(height,width,3),name="groud_truth")
    gd_half =Lambda(reshape_half)(gd)
    gd_qtr=Lambda(reshape_qtr)(gd)
    #ldr_reshape = Lambda(change_dim)(ldr_tensor)
    # conv block
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder1_conv1',use_bias=True)(ldr_tensor)
    x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder1_conv2',use_bias=True)(x)
    x=Conv2D(64,(3,3),strides=(2,2),activation='relu',padding='same',name='block1_conv3',use_bias=True)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='encoder2_conv1',use_bias=True)(x)
    x_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='encoder2_conv2',use_bias=True)(x)
    x= Conv2D(128, (3, 3),strides=(2,2), activation='relu', padding='same', name='block2_conv3',use_bias=True)(x_2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='encoder3_conv1',use_bias=True)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='encoder3_conv2',use_bias=True)(x)
    x_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='encoder3_conv3',use_bias=True)(x)
    x = Conv2D(256, (3, 3), strides=(2,2),activation='relu', padding='same', name='block3_conv4',use_bias=True)(x_3)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',use_bias=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',use_bias=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',use_bias=True)(x)
    #deconv block
    #block5

    x=Conv2DTranspose(256, (3, 3), strides=(2,2), activation='relu', padding='same', name='block5_deconv1',use_bias=True)(x)
    x = Lambda(cat)([x, x_3])
    x = Conv2DTranspose(256, (3, 3),  activation='relu', padding='same', name='decoder5_deconv2',use_bias=True)(x)
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='decoder5_deconv3',use_bias=True)(x)
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='decoder5_deconv4',use_bias=True)(x)
    x_qtr = Conv2D(3, (1, 1), activation='relu', padding='same', name='output_qtr', use_bias=True)(x)
    ssim_qtr=Lambda(SSIM)([x_qtr,gd_qtr])
    x=Lambda(cat)([x,x_qtr])
    #block6

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name='decoder6_deconv1',
                        use_bias=True)(x)
    x = Lambda(cat)([x, x_2])
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='decoder6_deconv2',
                        use_bias=True)(x)
    x = Conv2DTranspose(128, (3, 3),activation='relu', padding='same', name='decoder6_deconv3',
                        use_bias=True)(x)
    x_half = Conv2D(3, (3, 3), activation='relu', padding='same', name='output_half', use_bias=True)(x)
    ssim_half = Lambda(SSIM)([x_half, gd_half])
    x=Lambda(cat)([x,x_half])
    #block7

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='decoder7_deconv1',
                        use_bias=True)(x)
    x = Lambda(cat)([x, x_1])
    x = Conv2DTranspose(64, (3, 3),  activation='relu', padding='same', name='decoder7_deconv2',
                        use_bias=True)(x)
    x = Conv2DTranspose(64, (3, 3),  activation='relu', padding='same', name='decoder7_deconv3',
                        use_bias=True)(x)
    x=Conv2D(3, (1, 1), activation='sigmoid', padding='same', name='block8_conv1',use_bias=True)(x)
    ssim_origin=Lambda(SSIM)([x,gd])

    return Model(inputs=[ldr_tensor,gd],outputs=[ssim_qtr,ssim_half,ssim_origin])
def MEF_SSIM_compute(LDRseq, Fused, patch_size=8, adaptive_lum=True, num_seq=3):
    """
    计算多曝光图像融合评价
    :param LDRseq: 5 dims [patchsize, num_seq. height, width, color_channels]
    :param Fused: [patchsize, height, width, color_channels]
    :param patch_size: or window size, the area of mean or var compute
    :param adaptive_lum:
    :param num_seq:
    :param flag_keras: 是否返回更改维度
    :return:
    """

    """parameters"""
    mu_c = 0.5
    l_c = 0.5
    sigma_g = 0.2
    sigma_l = 0.2
    K1 = 0.01
    K2 = 0.03
    L = 1  # 8-bit for 255
    """compute parmeters"""
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    """compute with input LDR squence"""
    gMu_seq = K.tf.reduce_mean(LDRseq, [2, 3, 4], keep_dims=True)
    muX_seq = K.tf.reduce_mean(K.tf.nn.avg_pool3d(input=LDRseq,
                                                  ksize=[1, 1, patch_size, patch_size, 1],
                                                  strides=[1, 1, 1, 1, 1],
                                                  padding='SAME', data_format='NCDHW'),
                               axis=-1, keep_dims=True)
    muX_sq_seq = muX_seq * muX_seq
    sigmaX_sq_seq = K.tf.reduce_mean(K.tf.nn.avg_pool3d(input=LDRseq * LDRseq,
                                                        ksize=[1, 1, patch_size, patch_size, 1],
                                                        strides=[1, 1, 1, 1, 1],
                                                        padding='SAME', data_format='NCDHW'),
                                     axis=-1, keep_dims=True) - muX_sq_seq
    # sigmaX_sq = K.tf.reduce_max(sigmaX_sq_seq, 1, keep_dims=True)
    patch_index = K.tf.cast(K.tf.expand_dims(K.tf.argmax(sigmaX_sq_seq, axis=1), axis=1), K.tf.uint8)
    patch_index_transformed = K.tf.concat([K.tf.equal(patch_index, k) for k in range(num_seq)], axis=1)
    if adaptive_lum is True:
        LX_seq = K.tf.exp(- K.tf.divide((muX_seq - l_c) ** 2, 2 * sigma_l ** 2)
                          - K.tf.divide((gMu_seq - mu_c) ** 2, 2 * sigma_g ** 2))
        LX_normed = K.tf.divide(LX_seq, K.tf.reduce_sum(LX_seq, axis=1, keep_dims=True))
        muX = K.tf.reduce_sum(muX_seq * LX_normed, axis=1, keep_dims=True)
    else:
        muX = K.tf.reduce_sum(muX_seq * patch_index_transformed, axis=1, keep_dims=True)
    muX_sq = muX * muX
    """compute with Fused image"""
    muY = K.tf.expand_dims(K.tf.reduce_mean(K.tf.nn.avg_pool(value=Fused,
                                                             ksize=[1, patch_size, patch_size, 1],
                                                             strides=[1, 1, 1, 1],
                                                             padding='SAME', data_format='NHWC'),
                                            axis=-1, keep_dims=True),
                           axis=1)
    muY_sq = muY * muY
    sigmaY_sq = K.tf.expand_dims(K.tf.reduce_mean(K.tf.nn.avg_pool(value=Fused * Fused,
                                                                   ksize=[1, patch_size, patch_size, 1],
                                                                   strides=[1, 1, 1, 1],
                                                                   padding='SAME', data_format='NHWC'),
                                                  axis=-1, keep_dims=True),
                                 axis=1) - muY_sq
    """compute the cross of input LDR sequence and Fused"""
    Fused = K.tf.expand_dims(Fused, 1)
    muXY_seq = K.tf.reduce_mean(K.tf.nn.avg_pool3d(input=LDRseq * Fused,
                                                   ksize=[1, 1, patch_size, patch_size, 1],
                                                   strides=[1, 1, 1, 1, 1],
                                                   padding='SAME', data_format='NCDHW'),
                                axis=-1, keep_dims=True)
    sigmaXY_seq = muXY_seq - muX_seq * muY
    # sigmaXY = K.tf.reduce_sum(sigmaXY_seq * K.tf.to_float(patch_index_transformed), axis=1, keep_dims=True)
    """compute the q-map"""
    if adaptive_lum:
        A1_patches = 2 * muX * muY + C1
        B1_patches = muX_sq + muY_sq + C1
    else:
        A1_patches = 2 * muX_seq * muY + C1
        B1_patches = muX_sq_seq + muY_sq + C1
    A2_patches = 2 * sigmaXY_seq + C2
    B2_patches = sigmaX_sq_seq + sigmaY_sq + C2
    qmap_seq = K.tf.divide(A1_patches * A2_patches, B1_patches * B2_patches, name='qmap_seq')
    qmap = K.tf.reduce_sum(qmap_seq * K.tf.to_float(patch_index_transformed), axis=1, keep_dims=True, name='qmap')
    Q = K.tf.reduce_mean(qmap, axis=[2, 3], name='Q')
    return [qmap, qmap_seq, Q]
def GetPatches(images,hdr, patchSize=800, stride=180):
    """
    把图片切分为patches
    :param images: 待切分图像
    :param hdr: 对应hdr
    :param patchSize: 切片大小
    :param stride: 切割步长，覆盖率为patchSize-stride
    :return: 切片
    """
    h, w, c = np.shape(images)[1],np.shape(images)[2],np.shape(images)[3]
    #assert(np.shape(images)[1::] == np.shape(hdr))
    input = []
    GT = []
    for i, j in product(range(0, h - stride, stride), range(0, w - stride, stride)):
        flag_i = i + patchSize > h
        flag_j = j + patchSize > w
        if flag_i and flag_j:
            input.append(images[:, - patchSize:, - patchSize:, :])
            GT.append(hdr[- patchSize:, - patchSize:, :])
        elif flag_i:
            input.append(images[:, - patchSize:, j: j + patchSize:, :])
            GT.append(hdr[- patchSize:, j: j + patchSize, :])
        elif flag_j:
            input.append(images[:, i: i + patchSize, - patchSize:, :])
            GT.append(hdr[i: i + patchSize, - patchSize:, :])
        else:
            GT.append(hdr[i: i + patchSize, j: j + patchSize, :])
            input.append(images[:, i: i + patchSize, j: j + patchSize, :])
    return np.squeeze(np.concatenate(np.split(np.stack(input),3,1),-1)),np.stack(GT)

def GetPatches2(images, patchSize=200, stride=180):
    """
    把图片切分为patches
    :param images: 待切分图像
    :param hdr: 对应hdr
    :param patchSize: 切片大小
    :param stride: 切割步长，覆盖率为patchSize-stride
    :return: 切片
    """
    h, w, c = np.shape(images)[1],np.shape(images)[2],np.shape(images)[3]
    #assert(np.shape(images)[1::] == np.shape(hdr))
    input = []
    GT = []
    for i, j in product(range(0, h - stride, stride), range(0, w - stride, stride)):
        flag_i = i + patchSize > h
        flag_j = j + patchSize > w
        if flag_i and flag_j:
            input.append(images[:, - patchSize:, - patchSize:, :])
            #GT.append(hdr[- patchSize:, - patchSize:, :])
        elif flag_i:
            input.append(images[:, - patchSize:, j: j + patchSize:, :])
            #GT.append(hdr[- patchSize:, j: j + patchSize, :])
        elif flag_j:
            input.append(images[:, i: i + patchSize, - patchSize:, :])
            #GT.append(hdr[i: i + patchSize, - patchSize:, :])
        else:
            #GT.append(hdr[i: i + patchSize, j: j + patchSize, :])
            input.append(images[:, i: i + patchSize, j: j + patchSize, :])
    return np.squeeze(np.concatenate(np.split(np.stack(input),3,1),-1))

def load_data(data_path):
        data_list = []
        gd_list=[]
        dirs = os.listdir(data_path)
        for dir in dirs:
            abs_dir = os.path.join(data_path, dir)
            if (os.path.isdir(abs_dir)):
                reference_path = os.path.join(abs_dir, "reference")
                img_path = glob.glob(os.path.join(reference_path, "*.tif"))
                ue_image = img_as_float32(io.imread(img_path[0]))
                ne_image = img_as_float32(io.imread(img_path[1]))
                oe_image = img_as_float32(io.imread(img_path[2]))
                fused_path=glob.glob(os.path.join(reference_path, "*_15.tif"))
                fused_image = img_as_float32(io.imread(fused_path[0]))
                images = np.stack((ue_image, ne_image, oe_image))
                [input_patches,GT_patches]= GetPatches(images,fused_image)
                data_list.append(input_patches)
                gd_list.append(GT_patches)
        return np.concatenate(data_list),np.concatenate(gd_list)

def train_model(result_path):
    X, gd = load_data(result_path)
    model = vgg_16(800, 800)
    plot_model(model, to_file='./model.png')

    Y_train=np.ones(X.shape[0]*0.8)
    Y_test=np.ones(X.shape[0]*0.2)

    sgd = SGD(lr=0.001, decay=1e-10, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()
    with open("./record.json", "w") as dump_f:
        json.dump(model.to_json(), dump_f)
    # split train and test data

    X_train, X_test, gd_train, gd_test = train_test_split(
        X, gd, test_size=0.2, random_state=2)
    # input data to model and train

    history = model.fit([X_train,gd_train],Y_train, batch_size=2, epochs=10,
                        validation_data=([X_test,gd_test],Y_test ), verbose=1, shuffle=True)

    # evaluate the model
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    model.save_weights("./my_model.h5")
    file = open('./history.pkl', 'wb')
    pickle.dump(history.history, file)
    file.close()
    fig = plt.figure()  # 新建一张图
    plt.plot(history.history['acc'], label='training acc')
    plt.plot(history.history['val_acc'], label='val acc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    fig.savefig('./VGG16+Unet_acc.png')
    fig = plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('./VGG16+Unet_loss.png')








#
# def train_model():
#     train_labels = []
#     train_data_list=[]
#     for i in range(5):
#         with open("./cifar-10-batches-py/data_batch_" + str(i + 1), "rb") as fo:
#             dict = pickle.load(fo, encoding="bytes")
#             train_labels += dict[b'labels']
#             sub_data=np.reshape(dict[b'data'],(10000,32,32,3))
#             train_data_list.append(sub_data)
#     train_data=np.concatenate(train_data_list)
#
#     train_data,val_data,train_label,val_label=train_test_split(train_data,train_labels,test_size=0.2, random_state=2)
#
#     # with open("./cifar-10-batches-py/test_batch") as fo:
#     #     test_labels=dict[b'label']
#     #     dict = pickle.load(fo, encoding="bytes")
#     #     labels += dict[b'label']
#     #     test_data = np.reshape(dict[b'data'], (np.shape[0], 32, 32, 3))
#
#     model=vgg_16((32,32,3))
#     sgd = SGD(lr=0.001, decay=1e-10, momentum=0.9, nesterov=True)
#     model.compile(optimizer=sgd,
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     history = model.fit(train_data, train_label, batch_size=20, epochs=10,
#                         validation_data=(val_data, val_label), verbose=1, shuffle=True)
#
#     # evaluate the model


if __name__=="__main__":
  train_model("D:\Train_OnlyGT")


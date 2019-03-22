import numpy as np
from skimage import io, img_as_float
import glob
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.layers import Layer
import keras.backend as K
import keras.layers as KL

class MEF_SSIM(Layer):
    def __init__(self, patch_size=8, adaptive_lum=True, num_seq=3, change_dims=True, **kwargs):
        super(MEF_SSIM, self).__init__(**kwargs)
        self.trainable = False
        """parameters"""
        self.mu_c = 0.5
        self.l_c = 0.5
        self.sigma_g = 0.2
        self.sigma_l = 0.2
        self.K1 = 0.01
        self.K2 = 0.03
        self.L = 1  # 8-bit for 255
        """compute parmeters"""
        self.C1 = (self.K1 * self.L) ** 2
        self.C2 = (self.K2 * self.L) ** 2

        self.patch_size = patch_size
        self.adaptive_lum = adaptive_lum
        self.num_seq = num_seq
        self.change_dims = change_dims

    def call(self,
             inputs,
             # LDRseq, Fused,
             output_shape=None, name=None):
        """inputs"""
        LDR, Fused = inputs[0], inputs[1]
        """compute mef ssim"""
        qmap, qmap_seq, Q = KL.Lambda(MEF_SSIM_transform,
                                      arguments={'patch_size': self.patch_size,
                                                 'adaptive_lum': self.adaptive_lum,
                                                 'num_seq': self.num_seq})([LDR, Fused])
        if self.change_dims:
            """change dims"""
            qmap_seq = KL.Lambda(lambda x: K.squeeze(K.permute_dimensions(x, [0, 2, 3, 1, 4]), axis=-1),
                                 name='qmap_seq')(qmap_seq)
            qmap = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, axis=1), axis=-1), name='qmap')(qmap)
            Q = KL.Lambda(lambda x: K.reshape(x, [-1, 1]), name='Q')(Q)
        return [qmap, qmap_seq, Q]

    def compute_output_shape(self, input_shape):
        """
        计算输出张量维度
        :param input_shape:
        input_shape[0] ： LDRseq [None, 3, None, None, 3]
        input_shape[1] ： Fused [None, None, None, 3]
        :return:
        """
        LDR = input_shape[0]
        LDR_shape = [dim if dim is not None else None for dim in LDR]
        [p, n, h, w, c] = LDR_shape
        if self.change_dims:
            qmap = tuple([p, h, w])
            qmap_seq = tuple([p, h, w, n])
            Q = tuple([p, 1])
        else:
            qmap = tuple([p, 1, h, w, 1])
            qmap_seq = tuple([p, h, w, n, 1])
            Q = tuple([p, 1, 1])
        return [qmap, qmap_seq, Q]

def MEF_SSIM_transform(inputs, patch_size=8, adaptive_lum=True, num_seq=3):
    return MEF_SSIM_compute(inputs[0], inputs[1], patch_size=patch_size, adaptive_lum=adaptive_lum, num_seq=num_seq)

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

def See(vars, subfig=False):
    with K.tf.Session() as sess:
        for k, var in enumerate(vars.keys()):
            t = var.eval()
            plt.figure(k + 1)
            n = np.shape(t)[1]
            print(vars[var])
            if n == 1:
                if np.shape(t)[-1] == 1:
                    if len(vars[var]) == 3:
                        plt.imshow(np.squeeze(t), cmap=plt.cm.jet, vmin=vars[var][1], vmax=vars[var][2])
                    else:
                        plt.imshow(np.squeeze(t), cmap=plt.cm.jet)
                    plt.colorbar()
                else:
                    plt.imshow(np.squeeze(t))
                print(np.min(t), np.max(t))
                plt.axis('off')
                plt.title(vars[var][0])
            else:
                if subfig:
                    for i, image in enumerate(np.squeeze(t)):
                        print(str(i), np.min(image), np.max(image))
                        plt.subplot(1, n, i + 1)
                        if np.shape(t)[-1] == 3:
                            plt.imshow(np.squeeze(image))
                        else:
                            if len(vars[var]) == 3:
                                plt.imshow(np.squeeze(image), cmap=plt.cm.jet, vmin=vars[var][1], vmax=vars[var][2])
                            else:
                                plt.imshow(np.squeeze(image), cmap=plt.cm.jet)
                            plt.colorbar()
                        plt.axis('off')
                        plt.title(str(i))
                    plt.suptitle(vars[var][0])
                else:
                    tt = np.squeeze(np.dstack(np.split(np.squeeze(t), n)))
                    plt.imshow(tt, cmap=plt.cm.jet, vmin=np.min(tt), vmax=np.max(tt))
                    plt.colorbar()
                    plt.title(vars[var][0])
    plt.show()
    return 1

def main():
    """parameters"""
    adaptive_lum = True
    patch_size = 8

    """input"""
    Source_images = '../data/Mask'
    F_image = '../data/Mask/fused/Mask_HDRsoft-Mertens07.png'

    """Read and transform"""
    files = glob.glob(os.path.join(Source_images, '*.png'))
    images = []
    for file in files:
        images.append(img_as_float(io.imread(file)))
    Fused = img_as_float(io.imread(F_image))

    temp = sorted([(idx, np.mean(image), image) for idx, image in enumerate(images)], key=lambda x: x[1])
    images = [x[-1] for x in temp]
    del temp, Source_images, F_image, file, files

    """Comput paramaters"""
    n = len(images)
    c = np.shape(Fused)[-1]

    LDRseq = np.stack(images)
    print(LDRseq.shape)

    """tensors"""
    LDRseq = K.tf.constant(value=LDRseq, dtype=K.tf.float32)
    Fused = K.tf.constant(value=Fused, dtype=K.tf.float32)
    LDRseq = K.tf.expand_dims(LDRseq, axis=0)
    Fused = K.tf.expand_dims(Fused, axis=0)

    qmap, qmap_seq, Q = MEF_SSIM_compute(LDRseq, Fused, patch_size=8, adaptive_lum=True)

    vars = {
        # muX_seq: ['muX_seq'],
        # sigmaX_sq_seq: ['sigmaX_sq_seq'],
        # sigmaX_sq: ['sigmaX_sq'],
        # patch_index: ['patch_index', 0, n - 1],
        # muX: ['muX'],
        # muY: ['muY'],
        # sigmaY_sq: ['sigmaY_sq'],
        # sigmaXY_seq: ['sigmaXY_seq'],
        # A1_patches: ['A1_patches'],
        # A2_patches: ['A2_patches'],
        # B1_patches: ['B1_patches'],
        # B2_patches: ['B2_patches'],
        qmap_seq: ['qmap_init'],
        qmap: ['qmap'],
    }
    See(vars)

if  __name__ == '__main__':
    main()




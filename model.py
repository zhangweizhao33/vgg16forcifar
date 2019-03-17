import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Model
import pickle
from sklearn.model_selection import train_test_split
import matplotlib as plt
from keras.optimizers import SGD
def vgg_16(input_shape):
    input_tensor=Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)
    return Model(inputs=[input_tensor],outputs=[x])






def train_model():
    train_labels = []
    train_data_list=[]
    for i in range(5):
        with open("./cifar-10-batches-py/data_batch_" + str(i + 1), "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
            train_labels += dict[b'labels']
            sub_data=np.reshape(dict[b'data'],(10000,32,32,3))
            train_data_list.append(sub_data)
    train_data=np.concatenate(train_data_list)

    train_data,val_data,train_label,val_label=train_test_split(train_data,train_labels,test_size=0.2, random_state=2)

    # with open("./cifar-10-batches-py/test_batch") as fo:
    #     test_labels=dict[b'label']
    #     dict = pickle.load(fo, encoding="bytes")
    #     labels += dict[b'label']
    #     test_data = np.reshape(dict[b'data'], (np.shape[0], 32, 32, 3))

    model=vgg_16((32,32,3))
    sgd = SGD(lr=0.001, decay=1e-10, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, train_label, batch_size=20, epochs=10,
                        validation_data=(val_data, val_label), verbose=1, shuffle=True)

    # evaluate the model
    loss, acc = model.evaluate(val_data, val_label, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    model.save_weights("./my_model.h5")
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__=="__main__":
    train_model()


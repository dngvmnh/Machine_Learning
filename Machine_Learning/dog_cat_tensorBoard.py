import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

DATADIR = "C:/Users/dngvm/Projects/Machine Learning/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
THRESHOLD = 0.75

def show_image():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category) 
        for img in os.listdir(path):  
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            # plt.imshow(img_array, cmap='gray')  
            plt.imshow(new_array, cmap='gray')  
            plt.show()  
            break  
        break  

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category) 

        for img in os.listdir(path): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                training_data.append([new_array, class_num])  
            except Exception as e: 
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

def train_analyse():
    # create_training_data()
    # print(len(training_data))

    # random.shuffle(training_data)

    # X = []
    # y = []

    # for features, label in training_data:
    #     X.append(features)
    #     y.append(label)

    # X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # y = np.array(y)

    # pickle_out = open("X.pickle","wb")
    # pickle.dump(X, pickle_out)
    # pickle_out.close()

    # pickle_out = open("y.pickle","wb")
    # pickle.dump(y, pickle_out)
    # pickle_out.close()

    NAME = "dog_cat_cnn-{}".format(int(time.time()))

    X = pickle.load(open("X.pickle","rb"))
    y = pickle.load(open("y.pickle","rb"))

    X = X/255.0

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))  # 256 for default, 64 for 64x2 CNN
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))   # 256 for default, 64 for 64x2
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))   # comment out for default, comment in for 64x2 CNN


    model.add(Dense(1))
    model.add(Activation('sigmoid')) 

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])    # epochs = 3 for default, = 10 for 64x2 CNN

    # model.save('predict_dog_cat')

def train_optimize() :
    # dense_layers = [0, 1, 2]
    # layer_sizes = [32, 64, 128]
    # conv_layers = [1, 2, 3]
    dense_layers = [0]
    layer_sizes = [128]
    conv_layers = [2]
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                print(NAME)

                X = pickle.load(open("X.pickle","rb"))
                y = pickle.load(open("y.pickle","rb"))

                X = X/255.0

                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for _ in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())
                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'],
                            )

                model.fit(X, y,
                        batch_size=32,
                        epochs=30,
                        validation_split=0.3,
                        callbacks=[tensorboard])
                
                # model.save('predict_dog_cat')


if __name__ == "__main__" :
    # create_training_data()
    # print(len(training_data))

    # train_analyse()
    # train_optimize()
    
# tensorboard --logdir=logs/
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)
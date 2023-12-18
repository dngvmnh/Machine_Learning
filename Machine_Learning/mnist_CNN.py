import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import random

def train() :
    mnist = tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train=tf.keras.utils.normalize(x_train,axis=1)
    x_test=tf.keras.utils.normalize(x_test,axis=1)
    model =tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    model.save('predict_handwritten_num')

def predict_handwritten_num() :
    mnist = tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train=tf.keras.utils.normalize(x_train,axis=1)
    x_test=tf.keras.utils.normalize(x_test,axis=1)

    # new_model=tf.keras.models.load_model('predict_handwritten_num')
    # new_model=tf.keras.models.load_model('mnisst_LSTM')
    new_model=tf.keras.models.load_model('mnisst_CuDNNLSTM')

    IMAGE_TEST = random.randint(0, len(x_train) - 1)

    predictions = new_model.predict([x_train])
    print(np.argmax(predictions[IMAGE_TEST]))
    plt.imshow(x_train[IMAGE_TEST], cmap = plt.cm.binary)
    plt.show()

if __name__ =='__main__' :
    # train()
    predict_handwritten_num()
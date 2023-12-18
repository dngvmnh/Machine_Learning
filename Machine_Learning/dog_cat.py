import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


DATADIR = "C:/Users/dngvm/Projects/Machine Learning/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
THRESHOLD = 0.75

training_data = []

def show_image() :
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

def create_training_data() :
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

def train() :
    create_training_data()
    print(len(training_data))

    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    pickle_out = open("X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    X = pickle.load(open("X.pickle","rb"))
    y = pickle.load(open("y.pickle","rb"))

    X = X/255.0

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  

    model.add(Dense(128))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=200, validation_split=0.2)

    model.save('predict_dog_cat')

def predict_dog_cat_pickle() :
    X = pickle.load(open("X.pickle","rb"))
    y = pickle.load(open("y.pickle","rb"))

    X = X/255.0

    random_index = random.randint(0, len(X) - 1)
    random_image = X[random_index]
    random_image = random_image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    new_model=tf.keras.models.load_model('predict_dog_cat')
    predictions = new_model.predict(random_image)
    
    predicted_class = "Cat" if (predictions[0][0]) >= THRESHOLD else "Dog"

    plt.imshow(random_image.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"Predicted Class: {predicted_class}")
    plt.show()

    # print(f"Predicted Class: {predicted_class}")

def prepare(filepath) :
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def predict_dog_cat_filepath() :
    image_folder_path = "C:/Users/dngvm/Projects/Machine Learning/TestImages"
    image_files = os.listdir(image_folder_path)
    random_image_filename = random.choice(image_files)
    image_filepath = os.path.join(image_folder_path, random_image_filename)
    model = tf.keras.models.load_model("predict_dog_cat")
    prediction = model.predict([prepare(image_filepath)])
    predicted_class = CATEGORIES[int(prediction[0][0])]
    img_array = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img_array, cmap='gray')
    plt.title(f"Predicted Class: {predicted_class}")
    plt.show()

    # print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__" :
    # create_training_data()
    # print(len(training_data))
    # train()
    
    # predict_dog_cat_pickle()
    predict_dog_cat_filepath()
    
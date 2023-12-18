import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def train_LSTM() :
    mnist = tf.keras.datasets.mnist  
    (x_train, y_train),(x_test, y_test) = mnist.load_data()  

    x_train = x_train/255.0
    x_test = x_test/255.0

    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, decay=1e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    model.fit(x_train,
            y_train,
            epochs=3,
            validation_data=(x_test, y_test))
    
    model.save('mnist_LSTM')
    
def train_CuDNNLSTM() :
    mnist = tf.keras.datasets.mnist  
    (x_train, y_train),(x_test, y_test) = mnist.load_data()  

    x_train = x_train/255.0
    x_test = x_test/255.0

    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128,))
    model.add(Dropout(0.1))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, decay=1e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    model.fit(x_train,
            y_train,
            epochs=3,
            validation_data=(x_test, y_test))
    
    model.save('mnisst_CuDNNLSTM')
    
if __name__ == '__main__' :
    train_LSTM()
    train_CuDNNLSTM()
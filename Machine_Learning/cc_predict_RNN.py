import pandas as pd
from sklearn import preprocessing
from collections import deque
from numpy import random
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

FUTURE_PERIOD_PREDICT = 3  
RATIO_TO_PREDICT = "BTC-USD"
SEQ_LEN = 60  
BATCH_SIZE = 32
EPOCHS = 20
ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def prepare_df() :

    main_df = pd.DataFrame()

    for ratio in ratios:

        dataset = f'crypto_data/{ratio}.csv' 

        df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', f"{ratio}_close", f"{ratio}_volume"])  

        # df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  
        # df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

        df.set_index("time", inplace=True) 
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]  

        if len(main_df)==0:  
            main_df = df  
        else:  
            main_df = main_df.join(df)

    main_df.ffill(inplace=True) 
    main_df.dropna(inplace=True)

    main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
    main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

    # VALUE = 10
    # VALUE = len(main_df)

    # for RATIO_TO_PREDICT in ratios :
    # for RATIO_TO_PREDICT in RATIO_TO_PREDICT :
        
    #     main_df[f'{RATIO_TO_PREDICT}_future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
    #     main_df[f'{RATIO_TO_PREDICT}_target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df[f'{RATIO_TO_PREDICT}_future']))
    
    # print(main_df.head(VALUE)) 
    # print(main_df.head()) 

    times = sorted(main_df.index.values)
    last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

    val_main_df = main_df[(main_df.index >= last_5pct)] 
    main_df = main_df[(main_df.index < last_5pct)]  

    return main_df, val_main_df

def preprocess_balance_df(df):

    df = df.drop("future", axis=1)  

    for col in df.columns: 
        if col != "target": 
            df[col] = df[col].pct_change()  
            df.dropna(inplace=True)  
            df[col] = preprocessing.scale(df[col].values)  

    df.dropna(inplace=True) 

    sequential_data = []  
    prev_days = deque(maxlen = SEQ_LEN) 

    for i in df.values:  
        prev_days.append([n for n in i[:-1]]) 
        if len(prev_days) == SEQ_LEN: 
            sequential_data.append([np.array(prev_days), i[-1]]) 

    random.shuffle(sequential_data)  

    buys = []
    sells = []  

    for seq, target in sequential_data:  
        if target == 0:  
            sells.append([seq, target]) 
        elif target == 1:  
            buys.append([seq, target])  

    random.shuffle(buys)  
    random.shuffle(sells) 

    lower = min(len(buys), len(sells)) 

    buys = buys[:lower] 
    sells = sells[:lower]  

    sequential_data = buys+sells 
    random.shuffle(sequential_data) 

    X = []
    y = []

    for seq, target in sequential_data:  
        X.append(seq) 
        y.append(target)  

    return np.array(X), y  

def train(train_x,train_y,val_x,val_y) :

    NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))


    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"  
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_x, val_y),
        callbacks=[tensorboard, checkpoint],
    )

    score = model.evaluate(val_x, val_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save("models/{}".format(NAME))

def predict(data):

    model = tf.keras.models.load_model('predict_cc')

    predictions = model.predict(data)

    predicted_labels = [1 if prediction[1] > 0.5 else 0 for prediction in predictions]

    print(predicted_labels)
    

if __name__ == '__main__' :
    main_df, val_main_df = prepare_df()
    train_x, train_y = preprocess_balance_df(main_df)
    # val_x, val_y = preprocess_balance_df(val_main_df)
    # train_x = np.array(train_x)
    # train_y = np.array(train_y)
    # val_x = np.array(val_x)
    # val_y = np.array(val_y)

    # train(train_x,train_y,val_x,val_y)

    predict(train_x)

    # print(f"train data: {len(train_x)} validation: {len(val_x)}")
    # print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
    # print(f"VALIDATION Dont buys: {val_y.count(0)}, buys: {val_y.count(1)}")


 
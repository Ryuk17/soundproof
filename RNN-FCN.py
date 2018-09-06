from keras.models import Model
from keras.layers import Dense, Input,Add,LSTM,Flatten
import scipy.io as sio
import numpy as np
from keras.layers import Conv1D, GlobalAveragePooling1D
from keras import backend as K
import keras

# load data
data = sio.loadmat('data.mat')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

data_dim = 40
timesteps = 50
num_classes = 2

input = Input(shape=(timesteps, data_dim))

# RNN
r_layer1 = LSTM(32, return_sequences=True)(input)
r_layer2 = LSTM(64,return_sequences=True)(r_layer1)
r_layer2 = Flatten()(r_layer2)

# CNN
c_layer1 = Conv1D(64, 3, activation='relu')(input)
c_layer1 = keras.layers.BatchNormalization()(c_layer1)

c_layer2 = Conv1D(128, 3, activation='relu')(c_layer1)
c_layer2 = keras.layers.BatchNormalization()(c_layer2)

c_layer3 = Conv1D(64, 3, activation='relu')(c_layer2)
c_layer3 = keras.layers.BatchNormalization()(c_layer3)

c_pool = GlobalAveragePooling1D()(c_layer3)

merge = keras.layers.concatenate([r_layer2, c_pool])
output = Dense(2, activation='softmax')(merge)
model = Model(inputs=[input], outputs=output)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=64, epochs=32,validation_data=(test_x, test_y))
score = model.evaluate(test_x, test_y, batch_size=64)
print(score)
model.save('RNN-FCN'.h5')

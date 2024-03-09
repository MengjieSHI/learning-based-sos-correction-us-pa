import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib
import matplotlib.pyplot as plt
matplotlib.interactive(False)
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model

from model import *
from data_loader import *


def train(data_input, data_target, input_shape=(128, 1024, 1), dropout=0.5, learning_rate=1e-4, momentum=0.9,
          decay=1e-5, batch_size=10, epochs=150, validation_split=0.2, shuffle=True, model_name='pureUSsimulations', model_flag='TF_train'):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'models/' + model_name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    if model_flag is 'TF_train':
        base_model = load_model(save_path+'./train.h5')
        base_model.trainable = False
        model = TF_block(base_model, input_shape=input_shape)

    else:
        # base model
        model = decoder_block(encoder_block(input_shape, dropout))
    # optimizer
    # lr scheduler is not used
    opt = SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-5)
    # opt = Adam(learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=RootMeanSquaredError())
    model.summary()
    # model training
    history = model.fit(data_input, data_target, batch_size, epochs, verbose=1, validation_split=validation_split, shuffle=True)

    # model saving
    model.save(save_path+'./%s.h5'%model_flag)
    # show train/valid rMSE loss
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('rMSE LOSS')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim([0, 200])
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    # load data
    data_input = extract_signals('test_data.mat','non_filtered_rf_normalized')
    data_target = extract_signals('test_data.mat','sos_map_d2')
    # train
    model_name = 'pureUSsimulations'
    train(data_input, data_target, input_shape=(128, 1024, 1), dropout=0.5, learning_rate=1e-4, momentum=0.9,
          decay=1e-5, batch_size=10, epochs=5, validation_split=0.2, shuffle=True, model_name=model_name, model_flag='TF_train')


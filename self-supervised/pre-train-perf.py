from __future__ import print_function
import boto
import os
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
from sklearn.metrics import confusion_matrix,plot_precision_recall_curve, plot_roc_curve
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.applications import Xception
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model


np.random.seed(15)  # for reproducibility


"""
CNN used to classify spectrograms of normal participants (0) or depressed
participants (1). Using Theano backend and Theano image_dim_ordering:
(# channels, # images, # rows, # cols)
(1, 3040, 513, 125)
"""


def preprocess(X_train, X_test,y_train, y_test):
    """
    Convert from float64 to float32 and normalize normalize to decibels
    relative to full scale (dBFS) for the 4 sec clip.
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train_pre=copy.copy(X_train)
    X_test_pre = copy.copy(X_test)

    X_train_1=[]
    X_train_2=[]
    X_test_1=[]
    X_test_2=[]
    Y_train=[]
    Y_test=[]

    for X in X_train:
        X[range(8), :, :] = X[np.random.permutation(range(8)), :, :]

    for x in range(X_train.shape[0]):
        for y in range(5):
            yr=random.randint(0,X_train.shape[0]-1)
            X_train_1.append(X_train[x])
            X_train_2.append(X_train_pre[yr])
            Y_train.append(int(y_train[x]==y_train[yr]))

    for X in X_test:
        X[range(8), :, :] = X[np.random.permutation(range(8)), :, :]

    for x in range(X_test.shape[0]):
        for y in range(5):
            yr = random.randint(0,X_test.shape[0]-1)
            X_test_1.append(X_test[x])
            X_test_2.append(X_test_pre[yr])
            Y_test.append(int(y_test[x]==y_test[yr]))

    X_train_1 = np.array(X_train_1)
    X_train_2 = np.array(X_train_2)
    X_test_1 = np.array(X_test_1)
    X_test_2 = np.array(X_test_2)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    print(Y_train[:20])

    X_train_1 = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train_1])
    X_train_2 = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train_2])
    X_test_1 = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test_1])
    X_test_2 = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test_2])
    return X_train_1, X_train_2, X_test_1, X_test_2, Y_train, Y_test


def prep_train_test(X_train, y_train, X_test, y_test, nb_classes):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print('Train on {} samples, validate on {}'.format(X_train.shape[0],
                                                       X_test.shape[0]))

    # normalize to dBfS
    X_train_1, X_train_2, X_test_1, X_test_2, Y_train, Y_test= preprocess(X_train, X_test,y_train,y_test)

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    return X_train_1, X_train_2, X_test_1, X_test_2, Y_train, Y_test


def keras_img_prep(X_train_1, X_train_2, X_test_1, X_test_2):
    """
    Reshape feature matrices for Keras' expexcted input dimensions.
    For 'th' (Theano) dim_order, the model expects dimensions:
    (# channels, # images, # rows, # cols).
    """
    X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[2], X_train_1.shape[3], X_train_1.shape[1])
    X_train_2 = X_train_2.reshape(X_train_2.shape[0], X_train_2.shape[2], X_train_2.shape[3], X_train_2.shape[1])
    X_test_1 = X_test_1.reshape(X_test_1.shape[0], X_test_1.shape[2], X_test_1.shape[3], X_test_1.shape[1])
    X_test_2 = X_test_2.reshape(X_test_2.shape[0], X_test_2.shape[2], X_test_2.shape[3], X_test_2.shape[1])

    return X_train_1, X_train_2, X_test_1, X_test_2



def cnn(X_train_1, X_train_2, X_test_1, X_test_2, y_train, y_test, batch_size,
        nb_classes, epochs):
    """
    The Convolutional Neural Net architecture for classifying the audio clips
    as normal (0) or depressed (1).
    """

    trainer_1=Input(shape=(X_train_1.shape[1], X_train_1.shape[2], X_train_1.shape[3]))
    trainer_2=Input(shape=(X_train_2.shape[1], X_train_2.shape[2], X_train_2.shape[3]))

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 3),padding='same', strides=(1, 3)))

    model.add(Conv2D(32, (1, 3), padding='same', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 3), padding='same', strides=(1, 3)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    trainers_1=model(trainer_1)
    trainers_2=model(trainer_2)

    merged_vector = keras.layers.concatenate([trainers_1, trainers_2], axis=-1)

    predictions = Dense(512, activation='relu')(merged_vector)
    model.add(Dropout(0.5))
    predictions=Dense(nb_classes, activation='softmax')(predictions)

    fin_model = Model([trainer_1, trainer_2], predictions)

    Model.load_weights(filepath='../models/weights-improvement-168-0.56.hdf5', self=fin_model)

    fin_model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    #early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')

    # Evaluate accuracy on test and train sets
    score_train = fin_model.evaluate([X_train_1,X_train_2], y_train, verbose=0)
    print('Train accuracy:', score_train[1])
    score_test = fin_model.evaluate([X_test_1,X_test_2],y_test, verbose=0)
    print('Test accuracy:', score_test[1])

    return fin_model


def model_performance(model, X_train_1, X_train_2, X_test_1, X_test_2, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = np.argmax(model.predict([X_test_1,X_test_2]),axis=1)
    y_train_pred = np.argmax(model.predict([X_train_1,X_train_2]),axis=1)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test[:, 1]

    # plot train/test loss and accuracy. saves files in working dir
    print('Saving plots...')

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, conf_matrix


def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D
    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])


if __name__ == '__main__':

    path = '/mnt/xlancefs/home/rtg99/data/npz/'

    X_train = np.load(path+'train_samples_fbank.npz')
    y_train = np.load(path+'train_labels_fbank.npz')
    X_test = np.load(path+'test_samples_fbank.npz')
    y_test = np.load(path+'test_labels_fbank.npz')

    X_train, y_train, X_test, y_test = \
        X_train['arr_0'], y_train['arr_0'], X_test['arr_0'], y_test['arr_0']

    # CNN parameters
    batch_size = 64
    nb_classes = 2
    epochs = 500

    # normalalize data and prep for Keras
    print('Processing images for Keras...')
    X_train_1, X_train_2,X_test_1,X_test_2, y_train, y_test = prep_train_test(X_train, y_train,
                                                       X_test, y_test,
                                                       nb_classes=nb_classes)

    # reshape image input for Keras
    # used Theano dim_ordering (th), (# chans, # images, # rows, # cols)
    X_train_1, X_train_2, X_test_1, X_test_2 = keras_img_prep(X_train_1, X_train_2, X_test_1, X_test_2)

    # run CNN
    print('Fitting model...')
    model= cnn(X_train_1, X_train_2, X_test_1, X_test_2, y_train, y_test, batch_size,
                         nb_classes, epochs)

    print(model.summary())

    # evaluate model
    print('Evaluating model...')
    y_train_pred, y_test_pred, conf_matrix = model_performance(model, X_train_1, X_train_2, X_test_1, X_test_2, y_train, y_test)

    # save model to locally
    #print('Saving model locally...')
    #model_name = '/mnt/xlancefs/home/rtg99/models/pretrain_{}_lpc.h5'.format('00')
    #model.save(model_name)

    # custom evaluation metrics
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}".format(f1_score))

from __future__ import print_function
import boto
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,plot_precision_recall_curve, plot_roc_curve
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import plot_model


np.random.seed(15)  # for reproducibility


"""
CNN used to classify spectrograms of normal participants (0) or depressed
participants (1). Using Theano backend and Theano image_dim_ordering:
(# channels, # images, # rows, # cols)
(1, 3040, 513, 125)
"""


def preprocess(X_train, X_test):
    """
    Convert from float64 to float32 and normalize normalize to decibels
    relative to full scale (dBFS) for the 4 sec clip.
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])
    return X_train, X_test


def prep_train_test(X_train, y_train, X_test, y_test, nb_classes):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print('Train on {} samples, validate on {}'.format(X_train.shape[0],
                                                       X_test.shape[0]))

    # normalize to dBfS
    X_train, X_test = preprocess(X_train, X_test)

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test


def keras_img_prep(X_train, X_test):
    """
    Reshape feature matrices for Keras' expexcted input dimensions.
    For 'th' (Theano) dim_order, the model expects dimensions:
    (# channels, # images, # rows, # cols).
    """
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],   X_test.shape[2], X_test.shape[3],X_test.shape[1])

    return X_train, X_test



def cnn(X_train, y_train, X_test, y_test, batch_size,
        nb_classes, epochs):
    """
    The Convolutional Neural Net architecture for classifying the audio clips
    as normal (0) or depressed (1).
    """

    trainer_1 = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),name='conv2d_input')

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', strides=1, activation='relu',name='conv2d'))
    model.add(MaxPooling2D(pool_size=(4, 3), padding='same', strides=(1, 3),name='max_pooling2d'))

    model.add(Conv2D(32, (1, 3), padding='same', strides=1, activation='relu',name='conv2d_1'))
    model.add(MaxPooling2D(pool_size=(1, 3), padding='same', strides=(1, 3),name='max_pooling2d_1'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu',name='dense'))
    model.add(Dropout(0.5))

    for layer in model.layers[:]:
        layer.trainable = False

    trainers_1 = model(trainer_1)

    predictions = Dense(512, activation='relu',name='dense_10')(trainers_1)
    predictions = Dense(512, activation='relu', name='dense_11')(predictions)
    model.add(Dropout(0.5))
    predictions = Dense(2, activation='softmax',name='dense_12')(predictions)

    fin_model = Model(trainer_1, predictions)

    Model.load_weights(filepath='../models/weights-improvement-415-0.57.hdf5', self=fin_model,by_name=True)

    optimizer=Adadelta(lr=2e-4)

    fin_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Evaluate accuracy on test and train sets
    score_train = fin_model.evaluate(X_train, y_train, verbose=0)
    print('Train accuracy:', score_train[1])
    score_test = fin_model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score_test[1])

    return fin_model


def model_performance(model, X_train, X_test, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    y_train_pred = np.argmax(model.predict(X_train), axis=1)

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

    X_train = np.load(path+'train_samples_lpc.npz')
    y_train = np.load(path+'train_labels_lpc.npz')
    X_test = np.load(path+'test_samples_lpc.npz')
    y_test = np.load(path+'test_labels_lpc.npz')

    X_train, y_train, X_test, y_test = \
        X_train['arr_0'], y_train['arr_0'], X_test['arr_0'], y_test['arr_0']

    # CNN parameters
    batch_size = 32
    nb_classes = 2
    epochs = 500

    # normalalize data and prep for Keras
    print('Processing images for Keras...')
    X_train, X_test, y_train, y_test = prep_train_test(X_train, y_train,
                                                       X_test, y_test,
                                                       nb_classes=nb_classes)

    # reshape image input for Keras
    # used Theano dim_ordering (th), (# chans, # images, # rows, # cols)
    X_train, X_test = keras_img_prep(X_train, X_test)

    # run CNN
    print('Fitting model...')
    model = cnn(X_train, y_train, X_test, y_test, batch_size,
                         nb_classes, epochs)

    print(model.summary())

    # evaluate model
    print('Evaluating model...')
    y_train_pred, y_test_pred, conf_matrix = model_performance(model, X_train, X_test, y_train, y_test)

    # save model to locally
    print('Saving model locally...')
    #model_name = '/mnt/xlancefs/home/rtg99/models/cnn_{}_lpc.h5'.format('02')
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
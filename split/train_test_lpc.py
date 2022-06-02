import pandas as pd
import numpy as np
import librosa
import os
import boto
import random

np.random.seed(15)  # for reproducibility


def rand_samp_train_test_split(npz_file_dir):
    # files in directory
    npz_files = os.listdir(npz_file_dir)

    dep_samps = [f for f in npz_files if f.startswith('D_lpc')]
    norm_samps = [f for f in npz_files if f.startswith('N_lpc')]
    # calculate how many samples to balance classes
    max_samples = min(len(dep_samps), len(norm_samps))

    # randomly select max participants from each class without replacement
    dep_select_samps = np.random.choice(dep_samps, size=max_samples,
                                        replace=False)
    norm_select_samps = np.random.choice(norm_samps, size=max_samples,
                                         replace=False)

    # randomly select n_samples_per_person (40 in the case of a crop width
    # of 125) from each of the participant lists

    # REFACTOR this code!
    test_size = 0.2
    num_test_samples = int(len(dep_select_samps) * test_size)

    train_samples = []
    for sample in dep_select_samps[:-num_test_samples]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                train_samples.append(data[key])
    for sample in norm_select_samps[:-num_test_samples]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                train_samples.append(data[key])
    train_labels = np.concatenate((np.ones(int(len(train_samples) / 2)),
                                   np.zeros(int(len(train_samples) / 2))))

    test_samples = []
    for sample in dep_select_samps[-num_test_samples:]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                test_samples.append(data[key])
    for sample in norm_select_samps[-num_test_samples:]:
        npz_file = npz_file_dir + '/' + sample
        with np.load(npz_file) as data:
            for key in data.keys():
                test_samples.append(data[key])
    test_labels = np.concatenate((np.ones(int(len(test_samples) / 2)),
                                   np.zeros(int(len(test_samples) / 2))))

    return np.array(train_samples), train_labels, np.array(test_samples), test_labels

if __name__ == '__main__':
    # random sample from particpants npz files to ensure class balance
    train_samples, train_labels, test_samples, test_labels = rand_samp_train_test_split('/mnt/xlancefs/home/rtg99/data/processed')


    # save as npz locally
    print("Saving npz file locally...")
    np.savez('/mnt/xlancefs/home/rtg99/data/npz/train_samples_lpc.npz', train_samples)
    np.savez('/mnt/xlancefs/home/rtg99/data/npz/train_labels_lpc.npz', train_labels)
    np.savez('/mnt/xlancefs/home/rtg99/data/npz/test_samples_lpc.npz', test_samples)
    np.savez('/mnt/xlancefs/home/rtg99/data/npz/test_labels_lpc.npz', test_labels)

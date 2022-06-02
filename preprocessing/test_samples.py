import pandas as pd
import numpy as np
import librosa
import os
import boto
import random

fft_len = 1024
win_hop = 256
win_len = 1024
window = 'hann'
n_mel = 128
n_mfcc = 13
dct_type = 4
norm = 'ortho'
center = False

np.random.seed(15)  # for reproducibility

# 载入音频，读取时间序列y和采样率sr
y, sr = librosa.load('P481_no_silence.wav')

# STFT
matrix, n_fft = librosa.spectrum._spectrogram(y, n_fft=fft_len, hop_length=win_hop, win_length=win_len,
                                                center=center, power=2.0, window=window)


clipped_mat_orig = matrix[:, (matrix.shape[1] % 8):]
clipped_mat_0=np.split(clipped_mat_orig,8,axis=1)
pre_samples=[0]*8
samples=[0]*10

for i in range(8):
    clipped_mat=clipped_mat_0[i][:, (clipped_mat_0[i].shape[1] % 172):]
    n_splits = clipped_mat.shape[1] / 172
    cropped_sample_ls = np.split(clipped_mat, n_splits, axis=1)

    # get random samples
    pre_samples[i] = random.sample(cropped_sample_ls, 10)

for i in range(10):
    sp=np.array(pre_samples)
    samples[i]=np.array(pre_samples)[:,i]
    sp2=np.array(samples)
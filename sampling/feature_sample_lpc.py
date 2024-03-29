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

def clpc(y, order):

    dtype = y.dtype.type
    ar_coeffs = np.zeros(order + 1, dtype=dtype)
    ar_coeffs[0] = dtype(1)
    ar_coeffs_prev = np.zeros(order + 1, dtype=dtype)
    ar_coeffs_prev[0] = dtype(1)

    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]

    den = np.dot(fwd_pred_error, fwd_pred_error) + np.dot(
        bwd_pred_error, bwd_pred_error
    )

    for i in range(order):
        #if den <= 0:
        #    raise FloatingPointError("numerical error, input ill-conditioned?")

        reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)

        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]

        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

        q = dtype(1) - reflect_coeff ** 2
        den = q * den - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs


def build_class_dictionaries(dir_name):
    depressed_dict = dict()
    normal_dict = dict()
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('no_silence.wav'):
                partic_id = int(file.split('_')[0][1:])
                if in_dev_split(partic_id):
                    wav_file = os.path.join(subdir, file)
                    # matrix representation of spectrogram
                    y, sr = librosa.load(wav_file)
                    # lpc
                    lpc = clpc(y[0:win_len], 16).reshape(-1, 1)
                    t = win_hop
                    while t + win_len <= y.shape[0]:
                        sub1 = clpc(y[t:t + win_len], 16)
                        sub2 = sub1.reshape(-1, 1)
                        lpc = np.append(lpc, sub2, axis=1)
                        t = t + win_hop
                    lpc = lpc[1:]

                    depressed = get_depression_label(partic_id)  # 1 if True
                    if depressed:
                        depressed_dict[partic_id] = lpc
                    elif not depressed:
                        normal_dict[partic_id] = lpc
    return depressed_dict, normal_dict


def in_dev_split(partic_id):
    return partic_id in set(df_dev['Participant_ID'].values)


def get_depression_label(partic_id):
    return df_dev.loc[df_dev['Participant_ID'] ==
                      partic_id]['PHQ8_Binary'].item()


def build_class_sample_dict(segmented_audio_dict, n_samples, crop_width):
    class_samples_dict = dict()
    for partic_id, clip_mat in segmented_audio_dict.items():
        samples = get_random_samples(clip_mat, n_samples, crop_width)
        class_samples_dict[partic_id] = samples
    return class_samples_dict


#crop_width固定，默认n_samples=10（每个人random十次）
def get_random_samples(matrix, n_samples, crop_width):
    # crop full spectrogram into segments of width = crop_width
    clipped_mat_orig = matrix[:, (matrix.shape[1] % 8):]
    clipped_mat_0=np.split(clipped_mat_orig,8,axis=1)
    pre_samples=[0]*8
    samples=[0]*n_samples

    for i in range(8):
        clipped_mat=clipped_mat_0[i][:, (clipped_mat_0[i].shape[1] % crop_width):]
        n_splits = clipped_mat.shape[1] / crop_width
        cropped_sample_ls = np.split(clipped_mat, n_splits, axis=1)

        # get random samples
        pre_samples[i] = random.sample(cropped_sample_ls, n_samples)

    for i in range(n_samples):
        samples[i]=np.array(pre_samples)[:,i]
    #改输出
    return samples


def create_sample_dicts(crop_width):
    # build dictionaries of participants and segmented audio matrix
    depressed_dict, normal_dict = build_class_dictionaries('/mnt/xlancefs/home/rtg99/data/interim')
    n_samples = 10

    depressed_samples = build_class_sample_dict(depressed_dict, n_samples,
                                                crop_width)
    normal_samples = build_class_sample_dict(normal_dict, n_samples,
                                             crop_width)

    for key, _ in depressed_samples.items():
        path = '/mnt/xlancefs/home/rtg99/data/processed/'
        filename = 'D_lpc{}.npz'.format(key)
        outfile = path + filename
        np.savez(outfile, *depressed_samples[key])

    for key, _ in normal_samples.items():
        path = '/mnt/xlancefs/home/rtg99/data/processed'
        filename = '/N_lpc{}.npz'.format(key)
        outfile = path + filename
        np.savez(outfile, *normal_samples[key])


if __name__ == '__main__':
    # 导入csv
    df_train = pd.read_csv('/mnt/xlancefs/home/rtg99/wwwdaicwoz/train_split_Depression_AVEC2017.csv')
    df_test = pd.read_csv('/mnt/xlancefs/home/rtg99/wwwdaicwoz/dev_split_Depression_AVEC2017.csv')
    df_dev = pd.concat([df_train, df_test], axis=0)

    create_sample_dicts(crop_width=172)

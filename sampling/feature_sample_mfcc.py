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


def build_class_dictionaries(dir_name):
    depressed_dict1 = dict()
    normal_dict1 = dict()
    depressed_dict2 = dict()
    normal_dict2 = dict()
    depressed_dict3 = dict()
    normal_dict3 = dict()
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('no_silence.wav'):
                partic_id = int(file.split('_')[0][1:])
                print(partic_id)
                if in_dev_split(partic_id):
                    wav_file = os.path.join(subdir, file)
                    # matrix representation of spectrogram
                    y, sr = librosa.load(wav_file)
                    spec_amp, n_fft = librosa.spectrum._spectrogram(y, n_fft=fft_len, hop_length=win_hop,
                                                                    win_length=win_len,
                                                                    center=center, power=2.0, window=window)
                    # low power
                    fbank = librosa.feature.melspectrogram(S=spec_amp, n_mels=n_mel)
                    fbank = librosa.power_to_db(fbank, top_db=None)

                    # mfcc
                    mfcc = librosa.feature.mfcc(S=fbank, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm)

                    depressed = get_depression_label(partic_id)  # 1 if True
                    if depressed:
                        depressed_dict2[partic_id] = mfcc
                    elif not depressed:
                        normal_dict2[partic_id] = mfcc
    return depressed_dict1, depressed_dict2, normal_dict1, normal_dict2


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
    depressed_dict1, depressed_dict2, normal_dict1, normal_dict2 = build_class_dictionaries('/mnt/xlancefs/home/rtg99/data/interim')
    n_samples = 10

    depressed_samples2 = build_class_sample_dict(depressed_dict2, n_samples,
                                                 crop_width)
    normal_samples2 = build_class_sample_dict(normal_dict2, n_samples,
                                             crop_width)

    for key, _ in depressed_samples2.items():
        path = '/mnt/xlancefs/home/rtg99/data/processed/'
        filename = 'D_mfcc{}.npz'.format(key)
        outfile = path + filename
        np.savez(outfile, *depressed_samples2[key])

    for key, _ in normal_samples2.items():
        path = '/mnt/xlancefs/home/rtg99/data/processed'
        filename = '/N_mfcc{}.npz'.format(key)
        outfile = path + filename
        np.savez(outfile, *normal_samples2[key])


if __name__ == '__main__':
    # 导入csv
    df_train = pd.read_csv('/mnt/xlancefs/home/rtg99/wwwdaicwoz/train_split_Depression_AVEC2017.csv')
    df_test = pd.read_csv('/mnt/xlancefs/home/rtg99/wwwdaicwoz/dev_split_Depression_AVEC2017.csv')
    df_dev = pd.concat([df_train, df_test], axis=0)

    create_sample_dicts(crop_width=172)

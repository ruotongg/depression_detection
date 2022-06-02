import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display

# 参数初始化
fft_len = 1024
win_hop = 256
win_len = 1024
window = 'hann'
n_mel = 128
n_mfcc = 13
dct_type = 4
norm = 'ortho'
center = False


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

# 载入音频，读取时间序列y和采样率sr
y, sr = librosa.load('P319_no_silence.wav')



# STFT
spec_amp, n_fft = librosa.spectrum._spectrogram(y, n_fft=fft_len, hop_length=win_hop, win_length=win_len,
                                                center=center, power=2.0, window=window)

# low power
fbank=librosa.feature.melspectrogram(S=spec_amp, n_mels=n_mel)
fbank=librosa.power_to_db(fbank, top_db=None)

# mfcc
mfcc=librosa.feature.mfcc(S=fbank, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm)


#lpc
lpc=clpc(y[0:win_len], 16).reshape(-1, 1)
t=win_hop
while t+win_len <= y.shape[0]:
    sub1=clpc(y[t:t+win_len], 16)
    sub2=sub1.reshape(-1,1)
    lpc=np.append(lpc,sub2,axis=1)
    t=t+win_hop
lpc=lpc[1:]


# 绘图
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,10), sharex=True)
img=librosa.display.specshow(librosa.amplitude_to_db(spec_amp), y_axis='linear', sr=sr, hop_length=win_hop,
                         x_axis='time', ax=ax[0])
img2=librosa.display.specshow(fbank, y_axis='log', sr=sr, hop_length=win_hop,
                         x_axis='time', ax=ax[1])
img3=librosa.display.specshow(mfcc, y_axis='log', sr=sr, hop_length=win_hop,
                         x_axis='time', ax=ax[2])
img3=librosa.display.specshow(lpc, y_axis='log', sr=sr, hop_length=win_hop,
                         x_axis='time', ax=ax[3])
ax[0].set(title='STFT spectrogram')
ax[0].label_outer()
ax[1].set(title='Log-frequency power spectrogram')
ax[1].label_outer()
ax[2].set(title='MFCC spectrogram')
ax[2].label_outer()

fig.colorbar(img, ax=ax[0], format="%+2.f dB")
fig.colorbar(img2, ax=ax[1], format="%+2.f dB")
fig.colorbar(img2, ax=ax[2], format="%+2.f dB")

plt.show()

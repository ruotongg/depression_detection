# Self-supervised Learning Project - Audio Feature Analysis for Depression Detection

*Date: May 2021*
## Background
This is my thesis project at SJTU working with Prof. Mengyue Wu. With the discussion and treatment about mental illnesses becoming a hot topic in the current medical field, diagnostic methods related to computers, especially artificial intelligence, have been heavily researched and used. However, in the field of machine learning, especially deep learning, the creation of good diagnostic models requires a large amount of data from positive samples (i.e., diseased samples), and the limitations of medical tasks pose difficulties for this approach. The aim of this project is to learn and train a limited speech dataset of depression using a self-supervised learning approach and apply the training results to a depression diagnosis task.

## Description
The main work of this project can be divided into three parts: firstly, the speech features commonly used for speech emotion analysis and mental illness diagnosis are extracted from the DAIC-WOZ dataset, and the speech features suitable for the self-supervised learning task are identified through the analysis of their spectrograms and preliminary screening based on the effect of using machine learning classification; secondly, a reasonable depression task and dataset are designed according to the characteristics of the auxiliary task, this paper adopts the way of comparing the similarity between the original features and the disrupted features, and training on the established convolutional neural network; finally, using the trained model in the previous step, Fine-tuning is performed on the new convolutional neural network for the depression diagnosis task, and good classification results are obtained on the basis of saving a lot of training time.

*Dependencies: Python 3.6.5, pandas 0.24.0, NumPy 1.18.5, librosa 0.8.0, scikit-learn 0.24.0, tensorflow 1.15.0, keras 2.3.1*
## Structure
```
├─images
│      cnn.png
│      pre-train.png
│      pretrain_1_fbank.png
│      pretrain_1_lpc.png
│      pretrain_2_fbank.png
│      pretrain_2_lpc.png
│      train_1_fbank.png
│      train_1_lpc.png
│      train_2_fbank.png
│      train_2_lpc.png
│
├─preprocessing
│      extract_from_zip.py
│      feature.py
│      librosa_test.py
│      segmentation.py
│      test_samples.py
│
├─sampling
│      feature_sample_lpc.py
│      feature_sample_mfcc.py
│      feature_sample_others.py
│      feature_sample_stft.py
│
├─self-supervised
│      cnn-perf.py
│      pre-train-perf.py
│
├─split
│      train_test_fbank.py
│      train_test_lpc.py
│      train_test_mfcc.py
│      train_test_stft.py
│
└─traditional
        cnn.py
        random-forest.py
```
### preprocessing
Pre-process DAIC-WOZ dataset and extract audio features: Linear Prediction Coefficients (LPC), Short-time Fourier Transform (STFT), Filter Bank, Mel-Frequency Cepstrum (MFCC).

``segmentation.py``: Silent segment removal: The information contained in the speech signal after removing the silent segment is more concentrated and compact, which makes it more suitable for speech analysis tasks than the sparse original speech signal.
### split
Re-split training/validating/testing set for each audio feature.
### sampling
The problem to be solved in this section is the unbalanced dataset and the inconsistency of the data. The final solution adopted is to resample the speech data after removing the silent segments.

We consider segmenting the whole speech into 8 segments (based on PHQ-9 questionnaire), taking random 10 bars of speech features in each of these 8 segments (here we take 4 seconds), and combining these 8 segments into a three-dimensional feature matrix in order, which is used in machine learning classification and training.
### traditional
Traditional machine learning strategies for comparison & audio feature screening. We chose Random Forest (RF) & CNN.
### self-supervised
Core application. The auxiliary task we chose was to compare the similarity between two sets of features, and a binary classification task was designed; in terms of model design, for the convenience of Fine-tuning afterwards, the metamodel was still adopted similar to the previous convolutional neural network, on which a shared layer was designed to accomplish the similarity metric.

```pre-train-perf.py```: auxiliary task model

![Auxiliary task model](image/pre-train-model.png)

```cnn-perf.py```: self-supervised learning model

![Self-supervised learning model](image/cnn-model.png)
### images
Image output in this project.
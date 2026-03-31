Code cells:
========================================
!pip install librosa
# --- END OF CELL ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt  
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
# --- END OF CELL ---
def load_audio_files(path, label):
    audio_files = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            file_path = os.path.join(path, filename)
            audio, sample_rate = librosa.load(file_path, sr=16000)
            audio_files.append(audio)
            labels.append(label)
    return audio_files, labels, sample_rate

chopping_audio, chopping_labels, sample_rate = load_audio_files(chopping_path, label=0)
door_audio, door_labels, sample_rate = load_audio_files(door_path, label=1)
frying_audio, frying_labels, sample_rate = load_audio_files(frying_path, label=2)
gas_audio, gas_labels, sample_rate = load_audio_files(gas_path, label=3)
water_audio, water_labels, sample_rate = load_audio_files(water_path, label=4)
nothing_audio, nothing_labels, sample_rate = load_audio_files(nothing_path, label=5)
# --- END OF CELL ---
import numpy as np
from scipy.fftpack import dct

def extract_mfccs(audio, sample_rate, n_mfcc=13, n_mels=40, n_fft=2048, hop_length=512):
    pre_emphasis = 0.97
    emphasized_signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    frame_length, frame_step = n_fft, hop_length
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((n_fft + 1) * hz_points / sample_rate)

    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        for k in range(int(bin[m-1]), int(bin[m])): fbank[m-1, k] = (k - bin[m-1]) / (bin[m] - bin[m-1])
        for k in range(int(bin[m]), int(bin[m+1])): fbank[m-1, k] = (bin[m+1] - k) / (bin[m+1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks) # dB scale

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return np.mean(mfcc, axis=0)
# --- END OF CELL ---
n_mfcc = 13
mfcc_labels = [f'MFCC_{i+1}' for i in range(n_mfcc)]
feature_names = mfcc_labels

feature_names
# --- END OF CELL ---
from scipy.stats.distributions import f
def extract_features(audio_data, sample_rate):
    features = []
    for audio in audio_data:
        mfccs = extract_mfccs(audio, sample_rate)
        all_features = np.concatenate([mfccs])
        features.append(all_features)
    return np.array(features)

chopping_features = extract_features(chopping_audio, sample_rate)
door_features = extract_features(door_audio, sample_rate)
frying_features = extract_features(frying_audio, sample_rate)
gas_features = extract_features(gas_audio, sample_rate)
water_features = extract_features(water_audio, sample_rate)
nothing_features = extract_features(nothing_audio, sample_rate)

# --- END OF CELL ---
def extract_mfccs_2d(audio, sample_rate, n_mfcc=13, n_mels=40, n_fft=2048, hop_length=512):
    pre_emphasis = 0.97
    emphasized_signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    frame_length, frame_step = n_fft, hop_length
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((n_fft + 1) * hz_points / sample_rate)
    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        for k in range(int(bin[m-1]), int(bin[m])): fbank[m-1, k] = (k - bin[m-1]) / (bin[m] - bin[m-1])
        for k in range(int(bin[m]), int(bin[m+1])): fbank[m-1, k] = (bin[m+1] - k) / (bin[m+1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    max_len = 400
    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc

def extract_features_2d(audio_data, sample_rate):
    features = []
    for audio in audio_data:
        mfccs = extract_mfccs_2d(audio, sample_rate)
        features.append(mfccs)
    return np.array(features)

chopping_features_2d = extract_features_2d(chopping_audio, sample_rate)
door_features_2d = extract_features_2d(door_audio, sample_rate)
frying_features_2d = extract_features_2d(frying_audio, sample_rate)
gas_features_2d = extract_features_2d(gas_audio, sample_rate)
water_features_2d = extract_features_2d(water_audio, sample_rate)
nothing_features_2d = extract_features_2d(nothing_audio, sample_rate)

X_2d = np.concatenate((chopping_features_2d, door_features_2d, frying_features_2d, gas_features_2d, water_features_2d, nothing_features_2d))
Y_2d = np.array(chopping_labels + door_labels + frying_labels + gas_labels + water_labels + nothing_labels)
print(f"New dataset shape: {X_2d.shape}")
# --- END OF CELL ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()

y_pred_2d = np.argmax(model_2d.predict(X_test_reshaped), axis=1)
cm_2d = confusion_matrix(y_test, y_pred_2d)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_2d, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Chopping', 'Door', 'Frying', 'Gas', 'Water', 'Nothing'],
            yticklabels=['Chopping', 'Door', 'Frying', 'Gas', 'Water', 'Nothing'])
plt.title('Confusion Matrix - Custom 2D MFCC Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(classification_report(y_test, y_pred_2d, target_names=['Chopping', 'Door', 'Frying', 'Gas', 'Water', 'Nothing']))
# --- END OF CELL ---

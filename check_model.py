import os
import re
import numpy as np

# 1. Parse model.h
model_h_path = r"c:\Users\Administrator\Desktop\Train\Kitchenconfigure\kitchen\src\model.h"
tflite_path = r"c:\Users\Administrator\Desktop\Train\Kitchenconfigure\model.tflite"

with open(model_h_path, "r", encoding="utf-8") as f:
    text = f.read()

# Extract hex array
match = re.search(r'const\s+unsigned\s+char\s+g_model\s*\[.*?\]\s*=\s*\{(.*?)\};', text, re.DOTALL)
if match:
    hex_data = match.group(1)
    # clean and split
    hex_data = hex_data.replace('\n', '').replace('\r', '').replace(' ', '')
    byte_strs = [b for b in hex_data.split(',') if b]
    byte_array = bytearray([int(b, 16) for b in byte_strs])
    
    with open(tflite_path, "wb") as f:
        f.write(byte_array)
    print("Exported model.h to model.tflite")
else:
    print("Could not parse model.h")
    exit(1)

# 2. Extract features using Librosa (simulate training logic)
import librosa
import tensorflow as tf

wav_path = r"c:\Users\Administrator\Desktop\Train\Kitchenconfigure\datatest\fry1.wav"
y, sr = librosa.load(wav_path, sr=16000, mono=True)

# Pad/truncate to 1 sec
if len(y) > 16000:
    y = y[:16000]
else:
    y = np.pad(y, (0, 16000 - len(y)), 'constant')

# Extract MFCC (Assuming standard 13, 40 mels, 512 n_fft, 256 hop_length)
# User's cpp does pre-emphasis 0.97
y_pre = librosa.effects.preemphasis(y, coef=0.97)

# User's cpp does power spectrum = (re*re + im*im)/n_fft. 
# Librosa's power matches if we just take mag**2, but ESP32's fft window is hamming.
S = librosa.feature.melspectrogram(y=y_pre, sr=16000, n_fft=512, hop_length=256, n_mels=40, window='hamming', center=False)
# Librosa mel spectrogram uses energy S. ESP32 uses 10*log10(sum)
mel_energy = 10.0 * np.log10(np.maximum(S, 1e-12))
# DCT
mfccs = librosa.feature.mfcc(S=mel_energy, n_mfcc=13, dct_type=2, norm='ortho')

# Mean over time
mfccs_mean = np.mean(mfccs, axis=1)

# Scaler
scaler_mean = np.array([-519.809761, -124.014621, -8.75431547, -11.1560142, -6.39907990, -6.98835764, -1.46064090, -1.35306821, 0.116819005, 1.35346398, 2.46014028, 0.625107874, 1.17569215], dtype=np.float32)
scaler_std = np.array([127.53809314, 42.29953552, 22.39669615, 14.25831977, 9.45419393, 10.26505097, 7.4753217, 7.64669517, 7.16634196, 4.83842783, 4.58333441, 4.89204154, 4.14121623], dtype=np.float32)

scaled_features = (mfccs_mean - scaler_mean) / scaler_std

print("Python Extracted Scaled Features:")
print(" ".join([f"{x:.2f}" for x in scaled_features]))

# 3. TFLite inference
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.array([scaled_features], dtype=np.float32)
input_data = np.reshape(input_data, input_details[0]['shape'])

# If model needs INT8
if input_details[0]['dtype'] == np.int8:
    scale, zero_point = input_details[0]['quantization']
    input_data = np.round(input_data / scale + zero_point).astype(np.int8)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

if output_details[0]['dtype'] == np.int8:
    scale, zero_point = output_details[0]['quantization']
    output_data = (output_data.astype(np.float32) - zero_point) * scale

print("Softmax:")
print(output_data)

labels = ["chopping", "door", "frying", "gas", "water", "nothing"]
best = np.argmax(output_data[0])
print(f"Detected: {labels[best]}")

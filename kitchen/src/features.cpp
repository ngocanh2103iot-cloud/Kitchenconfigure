#include "audio.h"
#include "config.h"
#include "fbank_matrix.h"
#include <arduinoFFT.h>
#include <math.h>

float emphasized_signal[num_samples];
float current_frame[n_fft];
static double fft_real[n_fft];
static double fft_imag[n_fft];
float pow_frame[n_fft / 2 + 1];
float mel_energies[n_mels];
static float hamming_window[n_fft];
float dct_matrix[n_mfcc][n_mels];

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

ArduinoFFT<double> FFT = ArduinoFFT<double>(fft_real, fft_imag, n_fft, (double)SAMPLE_RATE);

void features_init() {
    // Tính sẵn Hamming window
    for (int k = 0; k < n_fft; k++)
        hamming_window[k] = 0.54f - 0.46f * cosf(2.0f * M_PI * k / (n_fft - 1));

    // Tính sẵn ma trận DCT
    float factor = sqrtf(2.0f / n_mels);
    for (int k = 1; k <= n_mfcc; k++)
        for (int m = 0; m < n_mels; m++)
            dct_matrix[k - 1][m] = factor * cosf(M_PI * k * (2.0f * m + 1.0f) / (2.0f * n_mels));
}

void extract_mfccs(const float *audio, float *mfcc_out) {
    // Pre-emphasis
    emphasized_signal[0] = audio[0];
    for (int i = 1; i < num_samples; i++)
        emphasized_signal[i] = audio[i] - 0.97f * audio[i - 1];

    float mfcc_sum[n_mfcc] = {0};

    for (int f = 0; f < num_frames; f++) {
        // Framing + Zero-padding
        int start_sample = f * hop_length;
        for (int k = 0; k < n_fft; k++)
            current_frame[k] = (start_sample + k < num_samples) ? emphasized_signal[start_sample + k] : 0.0f;

        // Hamming window
        for (int k = 0; k < n_fft; k++)
            current_frame[k] *= hamming_window[k];

        // FFT + Power spectrum
        for (int k = 0; k < n_fft; k++) {
            fft_real[k] = (double)current_frame[k];
            fft_imag[k] = 0.0;
        }
        FFT.compute(FFTDirection::Forward);
        for (int k = 0; k <= n_fft / 2; k++) {
            float re = (float)fft_real[k], im = (float)fft_imag[k];
            pow_frame[k] = (re * re + im * im) / n_fft;
        }

        // Mel filterbank
        int weight_idx = 0;
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            int start = mel_starts[m], length = mel_lengths[m];
            for (int i = 0; i < length; i++)
                sum += pow_frame[start + i] * mel_weights[weight_idx + i];
            weight_idx += length;
            mel_energies[m] = 20.0f * log10f(sum < 1e-12f ? 1e-12f : sum);
        }

        // DCT
        for (int k = 0; k < n_mfcc; k++) {
            float sum = 0.0f;
            for (int m = 0; m < n_mels; m++)
                sum += mel_energies[m] * dct_matrix[k][m];
            mfcc_sum[k] += sum;
        }
    }

    // Averaging
    for (int k = 0; k < n_mfcc; k++)
        mfcc_out[k] = mfcc_sum[k] / num_frames;
}

void extract_features(float *features) {
    extract_mfccs(get_audio(), features);
}

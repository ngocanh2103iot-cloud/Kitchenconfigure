#include "audio.h"
#include "config.h"
#include "fbank_matrix.h"
#include "esp_dsp.h"
#include <math.h>
#include <Arduino.h>

float current_frame[n_fft];
static float fft_buffer[n_fft * 2] __attribute__((aligned(16))); 
float pow_frame[n_fft / 2 + 1];
float mel_energies[n_mels];
static float hamming_window[n_fft];
float dct_matrix[n_mfcc][n_mels];

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void features_init() {
    // Khởi tạo FFT của thư viện ESP-DSP
    dsps_fft2r_init_fc32(NULL, n_fft);

    // Tính sẵn Hamming window
    for (int k = 0; k < n_fft; k++)
        hamming_window[k] = 0.54f - 0.46f * cosf(2.0f * M_PI * k / (n_fft - 1));

    // --- ĐÃ SỬA LỖI 1: Lấy MFCC từ 0, và áp dụng chuẩn hóa (ortho norm) đúng chuẩn Librosa ---
    for (int k = 0; k < n_mfcc; k++) {
        // k = 0 dùng sqrt(1/N), k > 0 dùng sqrt(2/N)
        float factor = (k == 0) ? sqrtf(1.0f / n_mels) : sqrtf(2.0f / n_mels);
        
        for (int m = 0; m < n_mels; m++) {
            dct_matrix[k][m] = factor * cosf(M_PI * k * (2.0f * m + 1.0f) / (2.0f * n_mels));
        }
    }
}

void extract_mfccs(const int16_t *audio, float *mfcc_out) {
    float mfcc_sum[n_mfcc] = {0}; 

    // --- VÒNG LẶP CUỐN CHIẾU (FRAME-BY-FRAME) ---
    for (int f = 0; f < num_frames; f++) {
        
        int start_sample = f * hop_length;

        // A. Cắt khung và Pre-emphasis chuẩn hóa sang float
        for (int k = 0; k < n_fft; k++) {
            int idx = start_sample + k;
            if (idx < num_samples) {
                // Chuyển sang float và chuẩn hóa về [-1, 1]
                float current_val = (float)audio[idx] / 32768.0f;
                float prev_val = (idx > 0) ? (float)audio[idx - 1] / 32768.0f : current_val; 
                current_frame[k] = (current_val - 0.97f * prev_val) * hamming_window[k];
            } else {
                current_frame[k] = 0.0f;
            }
        }

        // B. FFT dùng ESP-DSP (Tiết kiệm RAM hơn arduinoFFT)
        for (int k = 0; k < n_fft; k++) {
            fft_buffer[k * 2] = current_frame[k];
            fft_buffer[k * 2 + 1] = 0.0f;
        }
        
        dsps_fft2r_fc32(fft_buffer, n_fft);
        dsps_bit_rev_fc32(fft_buffer, n_fft);

        // C. Tính Power Spectrum
        for (int k = 0; k <= n_fft / 2; k++) {
            float re = fft_buffer[k * 2];
            float im = fft_buffer[k * 2 + 1];
            pow_frame[k] = (re * re + im * im) / n_fft;
        }

        // D. Mel filterbank
        int weight_idx = 0;
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            int start = mel_starts[m], length = mel_lengths[m];
            
            if (length > 0) {
                // Dùng dot product tối ưu của ESP-DSP
                dsps_dotprod_f32(&pow_frame[start], &mel_weights[weight_idx], &sum, length);
                weight_idx += length;
            }

            if (sum < 1e-12f) sum = 1e-12f;
            
            // --- ĐÃ SỬA LỖI 2: Dùng 10.0f cho Power Spectrum (chuẩn Librosa) ---
            mel_energies[m] = 10.0f * log10f(sum);
        }

        // E. DCT
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

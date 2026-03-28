#include "audio.h"
#include "config.h"
#include "dsps_dotprod.h"
#include "esp_dsp.h"
#include "fbank_matrix.h"
#include <math.h>
#include <stdio.h>

float pre_emphasis = 0.97;
float emphasized_signal[num_samples];
float frames[num_frames][n_fft];
float fft_in[n_fft * 2];
float pow_frames[num_frames][n_fft / 2 + 1];
float dct_matrix[n_mfcc][n_mels];

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
void pre_emphasis_filter(const float *input, float *output, int length) {
  output[0] = input[0];
  for (int i = 1; i < length; i++) {
    output[i] = input[i] - pre_emphasis * input[i - 1];
  }
}

void features_init() {
  dsps_fft2r_init_fc32(NULL, n_fft);

  float factor = sqrtf(2.0f / n_mels);
  for (int k = 1; k <= n_mfcc; k++) {
    for (int m = 0; m < n_mels; m++) {
      dct_matrix[k - 1][m] =
          factor * cosf(M_PI * k * (2.0f * m + 1.0f) / (2.0f * n_mels));
    }
  }
}

void extract_mfccs(const float *audio, float *mfcc) {

  // Lọc pre-emphasis — tăng cường tần số cao
  pre_emphasis_filter(audio, emphasized_signal, num_samples);

  // Phân khung (Framing) + đệm 0 (Zero-padding)
  static float pad_signal[num_frames * hop_length + n_fft];
  for (int i = 0; i < num_samples; i++)
    pad_signal[i] = emphasized_signal[i];
  for (int i = num_samples; i < num_frames * hop_length + n_fft; i++)
    pad_signal[i] = 0;
  for (int f = 0; f < num_frames; f++)
    for (int k = 0; k < n_fft; k++)
      frames[f][k] = pad_signal[f * hop_length + k];

  // Cửa sổ Hamming — làm mềm biên khung, giảm nhiễu phổ
  for (int f = 0; f < num_frames; f++)
    for (int k = 0; k < n_fft; k++)
      frames[f][k] *= 0.54f - 0.46f * cosf(2.0f * M_PI * k / (n_fft - 1));

  // FFT + Power Spectrum — chuyển từ miền thời gian sang tần số
  for (int f = 0; f < num_frames; f++) {
    for (int k = 0; k < n_fft; k++) {
      fft_in[2 * k] = frames[f][k];
      fft_in[2 * k + 1] = 0.0f;
    }
    dsps_fft2r_fc32_ansi(fft_in, n_fft);
    dsps_bit_rev2r_fc32(fft_in, n_fft);

    for (int k = 0; k <= n_fft / 2; k++) {
      float re = fft_in[k * 2];
      float im = fft_in[k * 2 + 1];
      pow_frames[f][k] = (re * re + im * im) / n_fft;
    }
  }

  // Bước 5: Lọc Mel Filterbank (Dùng Sparse Matrix 1D siêu nhẹ ~1.8KB ROM)
  static float filter_banks[num_frames][n_mels];
  for (int f = 0; f < num_frames; f++) {
    int weight_idx = 0;
    for (int m = 0; m < n_mels; m++) {
      float sum = 0.0f;
      int start = mel_starts[m];
      int length = mel_lengths[m];

      if (length > 0) {
        dsps_dotprod_f32(&pow_frames[f][start], &mel_weights[weight_idx], &sum,
                         length);
        weight_idx += length;
      }
      if (sum < 1e-12f)
        sum = 1e-12f;
      filter_banks[f][m] = 20.0f * log10f(sum);
    }
  }

  //  Khai triển Cosine Rời rạc (DCT)
  static float mfccs_frames[num_frames][n_mfcc];
  for (int f = 0; f < num_frames; f++) {
    for (int k = 0; k < n_mfcc; k++) {
      float sum = 0.0f;
      dsps_dotprod_f32(&filter_banks[f][0], &dct_matrix[k][0], &sum, n_mels);
      mfccs_frames[f][k] = sum;
    }
  }

  //  Tính giá trị trung bình (Averaging) để ra 1D Array cuối cùng
  for (int k = 0; k < n_mfcc; k++) {
    float sum = 0.0f;
    for (int f = 0; f < num_frames; f++) {
      sum += mfccs_frames[f][k];
    }
    mfcc[k] = sum / num_frames;
  }
}

void extract_features(float *features) { extract_mfccs(get_audio(), features); }

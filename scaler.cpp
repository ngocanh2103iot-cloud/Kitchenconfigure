#include "scaler.h"

// Cập nhật với 13 giá trị Mean mới
float scaler_mean[13] = {
    -519.809761f, -124.014621f, -8.75431547f, -11.1560142f,
    -6.39907990f, -6.98835764f, -1.46064090f, -1.35306821f,
    0.116819005f, 1.35346398f, 2.46014028f, 0.625107874f,
    1.17569215f
};

// Cập nhật với 13 giá trị STD mới
float scaler_std[13] = {
    127.53809314f, 42.29953552f, 22.39669615f, 14.25831977f,
    9.45419393f, 10.26505097f, 7.4753217f, 7.64669517f,
    7.16634196f, 4.83842783f, 4.58333441f, 4.89204154f,
    4.14121623f
};

void scale_features(float *features) {
    // Vòng lặp được sửa từ 10 thành 13 để khớp với số lượng MFCC mới
    for (int i = 0; i < 13; i++) {
        features[i] = (features[i] - scaler_mean[i]) / scaler_std[i];
    }
}
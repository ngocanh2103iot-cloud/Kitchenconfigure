#include "features.h"
#include "audio.h"
#include "config.h"
#include <ArduinoFFT.h>
#include <math.h>

double real[FRAME_SIZE];
double imag[FRAME_SIZE];
ArduinoFFT<double> FFT = ArduinoFFT<double>(real, imag, FRAME_SIZE, SAMPLE_RATE);

void extract_features(float *features) {
    float* audio = get_audio();
    for(int i = 0; i < FRAME_SIZE; i++) {
        real[i] = (double)audio[i] * 2.0;
        imag[i] = 0;
    }

    // FFT
    FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();

    int bins = FRAME_SIZE / 2;
    float mfcc[13];

    //MFCC
    int group_size = bins / 13; 
    for(int i = 0; i < 13; i++) {
        float sum = 0;
        for(int j = i * group_size; j < (i + 1) * group_size; j++) {
            sum += (float)real[j];
        }
      
        if (sum < 1e-9f) sum = 1e-9f; 
        mfcc[i] = logf(sum); 
    }

    // Spectral Centroid 
    float sum_mag = 0, weighted_mag = 0;
    for(int i = 1; i < bins; i++) {
        float mag = (float)real[i];
        sum_mag += mag;
        weighted_mag += ((i * SAMPLE_RATE) / (float)FRAME_SIZE) * mag;
    }
    float centroid = (sum_mag > 0.001f) ? (weighted_mag / sum_mag) : 0;

    // Spectral Rolloff
    float threshold = 0.85f * sum_mag;
    float cumulative = 0, rolloff = 0;
    for(int i = 0; i < bins; i++) {
        cumulative += (float)real[i];
        if(cumulative >= threshold) {
            rolloff = (i * SAMPLE_RATE) / (float)FRAME_SIZE;
            break;
        }
    }

    features[0] = mfcc[5];
    features[1] = mfcc[6];
    features[2] = mfcc[7];
    features[3] = mfcc[12];
    features[4] = mfcc[11];
    features[5] = mfcc[9];
    features[6] = centroid;
    features[7] = mfcc[3];
    features[8] = mfcc[10];
    features[9] = rolloff;
}
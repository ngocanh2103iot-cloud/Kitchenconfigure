#pragma once

void features_init();                           
void extract_mfccs(const float *audio, float *mfcc); 
void extract_features(float *features);
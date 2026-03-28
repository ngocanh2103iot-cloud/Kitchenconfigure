#include <Arduino.h>
#include "config.h"
#include "audio.h"
#include "features.h"
#include "scaler.h"
#include "inference.h"

float features[FEATURE_COUNT];

void setup() {
    Serial.begin(115200);
    audio_init();
    model_init();
    Serial.println("System Started");
}

void loop() {
    record_audio();
    extract_features(features);
    scale_features(features); 
    
    int label = run_inference(features);
    print_result(label);
    
    Serial.println("--------------------------");
    delay(300); 
}
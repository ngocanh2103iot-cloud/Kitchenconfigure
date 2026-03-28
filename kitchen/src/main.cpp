#include "audio.h"
#include "config.h"
#include "features.h"
#include "inference.h"
#include "scaler.h"
#include <Arduino.h>


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
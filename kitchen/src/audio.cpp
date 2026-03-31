#include <Arduino.h>
#include "audio.h"
#include "config.h"
#include "dummy_audio.h"

static int16_t audio_buffer[num_samples]; 

void audio_init() {
    Serial.println("Audio Init (Dummy Mode)");
}

void record_audio() {
    // Copy dữ liệu từ mảng dummy_audio sang audio_buffer để giả lập việc ghi âm
    for(int i = 0; i < num_samples; i++) {
        audio_buffer[i] = dummy_audio[i];
    }
    
    // Giả lập thời gian chờ ghi âm thực tế (1 giây)
    delay(1000); 
}

int16_t* get_audio() {
    return audio_buffer;
}
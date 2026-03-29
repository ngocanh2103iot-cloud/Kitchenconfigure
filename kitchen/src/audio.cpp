#include <Arduino.h>
#include <driver/i2s.h>
#include "audio.h"
#include "config.h"

#ifdef USE_DUMMY_AUDIO
#include "dummy_audio.h"
#endif

static int16_t audio_buffer[num_samples]; 

void audio_init() {
#ifndef USE_DUMMY_AUDIO
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = 0,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
#else
    Serial.println("Audio: Using DUMMY AUDIO (Sample: Chopping)");
#endif
}

void record_audio() {
#ifndef USE_DUMMY_AUDIO
    size_t bytes_read;
    int32_t raw_samples[FRAME_SIZE]; 
    
    for(int offset = 0; offset < num_samples; offset += FRAME_SIZE) {
        i2s_read(I2S_NUM_0, raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);
        int samples_read = bytes_read / sizeof(int32_t);
        for (int i = 0; i < samples_read; i++) {
            if (offset + i < num_samples) {
                // Chuyển 32-bit sang 16-bit để tiết kiệm RAM
                audio_buffer[offset + i] = (int16_t)(raw_samples[i] >> 16); 
            }
        }
    }
#else
    // Copy dummy data to audio buffer, convert back to int16 range
    for (int i = 0; i < num_samples; i++) {
        audio_buffer[i] = (int16_t)(dummy_audio[i] * 32767.0f);
    }
    // Simulation delay
    delay(100); 
#endif
}

int16_t* get_audio() {
    return audio_buffer;
}
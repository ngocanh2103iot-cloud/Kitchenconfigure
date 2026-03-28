#include <Arduino.h>
#include <driver/i2s.h>
#include "audio.h"
#include "config.h"

static float audio_buffer[num_samples]; 

void audio_init() {
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
}

void record_audio() {
    size_t bytes_read;
    int32_t raw_samples[FRAME_SIZE]; 
    

    for(int offset = 0; offset < num_samples; offset += FRAME_SIZE) {
        i2s_read(I2S_NUM_0, raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);
        int samples_read = bytes_read / sizeof(int32_t);
        for (int i = 0; i < samples_read; i++) {
            if (offset + i < num_samples) {
                audio_buffer[offset + i] = (float)raw_samples[i] / 2147483648.0f; 
            }
        }
    }
}

float* get_audio() {
    return audio_buffer;
}
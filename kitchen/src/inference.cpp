#include "inference.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

namespace {
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
alignas(16) uint8_t tensorArena[15 * 1024];
} // namespace

const char *labels[6] = {"chopping", "door",  "frying",
                         "gas",      "water", "nothing"};

void model_init() {
  const tflite::Model *model = tflite::GetModel(g_model);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroErrorReporter errorReporter;

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, sizeof(tensorArena), &errorReporter);

  interpreter = &staticInterpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
}

int run_inference(float *features) {
  // 1. Đưa dữ liệu vào (Tự động chuyển sang INT8 nếu model yêu cầu)
  if (input->type == kTfLiteInt8) {
    for (int i = 0; i < 10; i++) {
      input->data.int8[i] = (int8_t)(features[i] / input->params.scale +
                                     input->params.zero_point);
    }
  } else {
    for (int i = 0; i < 10; i++) {
      input->data.f[i] = features[i];
    }
  }

  // In input để debug
  Serial.print("input: ");
  for (int i = 0; i < 10; i++) {
    Serial.print(features[i]);
    Serial.print(" ");
  }
  Serial.println();

  if (interpreter->Invoke() != kTfLiteOk)
    return -1;

  // 2. Đọc kết quả và tìm nhãn cao nhất
  int best_idx = 0;
  float max_score = -1.0e30f;

  Serial.print("softmax: ");
  for (int i = 0; i < 6; i++) {
    float score;
    if (output->type == kTfLiteInt8) {
      score = (output->data.int8[i] - output->params.zero_point) *
              output->params.scale;
    } else {
      score = output->data.f[i];
    }
    Serial.print(score);
    Serial.print(" ");

    if (score > max_score) {
      max_score = score;
      best_idx = i;
    }
  }
  Serial.println();
  return best_idx;
}

void print_result(int label) {
  if (label >= 0) {
    Serial.print("Detected: ");
    Serial.println(labels[label]);
  }
}
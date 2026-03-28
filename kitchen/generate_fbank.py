import numpy as np
import os

sample_rate = 16000
n_fft = 512
n_mels = 40

# Hàm tạo ma trận Mel
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
hz_points = (700 * (10**(mel_points / 2595) - 1))
bin_points = np.floor((n_fft + 1) * hz_points / sample_rate)

fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
for m in range(1, n_mels + 1):
    for k in range(int(bin_points[m-1]), int(bin_points[m])):
        fbank[m-1, k] = (k - bin_points[m-1]) / (bin_points[m] - bin_points[m-1])
    for k in range(int(bin_points[m]), int(bin_points[m+1])):
        fbank[m-1, k] = (bin_points[m+1] - k) / (bin_points[m+1] - bin_points[m])

# Rút gọn ma trận thành 3 mảng (Sparse Matrix 1D)
mel_starts = []
mel_lengths = []
mel_weights = []

for m in range(n_mels):
    non_zero = np.where(fbank[m] > 0)[0]
    if len(non_zero) > 0:
        start = non_zero[0]
        length = len(non_zero)
        mel_starts.append(int(start))
        mel_lengths.append(int(length))
        mel_weights.extend(fbank[m, start:start+length].tolist())
    else:
        mel_starts.append(0)
        mel_lengths.append(0)

# Tạo nội dung file header C++
header_content = f"""#pragma once

// Ma trận Mel Filterbank Nén (Sparse Matrix)
// Gồm 3 mảng 1 chiều giúp tiết kiệm RAM và ROM.
// Số lượng bộ lọc (n_mels): {n_mels}
// Tổng số phần tử trọng số khác 0: {len(mel_weights)} ({len(mel_weights) * 4} bytes)

const int mel_starts[{n_mels}] = {{ {", ".join(map(str, mel_starts))} }};
const int mel_lengths[{n_mels}] = {{ {", ".join(map(str, mel_lengths))} }};
const float mel_weights[{len(mel_weights)}] = {{
"""

# Ghi mel_weights cắt từng dòng cho đẹp
chunk_size = 8
for i in range(0, len(mel_weights), chunk_size):
    chunk = mel_weights[i:i+chunk_size]
    header_content += "  " + ", ".join([f"{val:.6f}f" for val in chunk]) + ",\n"

header_content += "};\n"

# Ghi ra file
output_path = os.path.join(os.path.dirname(__file__), "src", "fbank_matrix.h")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(header_content)

print(f"Generated sparse matrix successfully: {output_path}")
print(f"Total weights: {len(mel_weights)}, Size: {len(mel_weights) * 4} bytes (vs {n_mels * fbank.shape[1] * 4} bytes in dense matrix)")

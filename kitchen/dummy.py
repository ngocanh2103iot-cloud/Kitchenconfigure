import librosa
import numpy as np
import os

# 1. Tìm file chopping trong thư mục datatest
datatest_dir = os.path.join("..", "datatest")
wav_file = None

if os.path.exists(datatest_dir):
    # Tìm file có tên chứa "chopping"
    wav_files = [f for f in os.listdir(datatest_dir) if "chopping" in f.lower() and f.endswith(".wav")]
    if wav_files:
        wav_file = os.path.join(datatest_dir, wav_files[0])
        print(f"👉 Tìm thấy file mẫu: {wav_file}")
    else:
        print("❌ Không tìm thấy file mẫu 'chopping' nào trong thư mục datatest.")
        exit(1)
else:
    print("❌ Không tìm thấy thư mục datatest.")
    exit(1)

print(f"Đang đọc file {wav_file}...")
try:
    # Đọc file, tự động ép về Sample Rate 16000Hz, lấy đúng 1 giây
    y, sr = librosa.load(wav_file, sr=16000, duration=1.0)
except Exception as e:
    print(f"❌ Lỗi khi đọc file âm thanh: {e}")
    exit(1)

# Nếu file ngắn hơn 1 giây (< 16000 mẫu), đệm thêm số 0 cho đủ
if len(y) < 16000:
    y = np.pad(y, (0, 16000 - len(y)), 'constant')
else:
    y = y[:16000] # Nếu dài hơn thì cắt lấy đúng 16000

# 2. Bắt đầu tạo nội dung file C++ Header
header_content = """#pragma once
// File âm thanh giả lập 1 giây từ mẫu 'chopping' để test Model AI
const float dummy_audio[16000] = {
"""

print("Đang chuyển đổi sang mảng C++...")
# Chèn dữ liệu vào (cắt 10 số mỗi dòng cho dễ nhìn)
for i in range(0, len(y), 10):
    chunk = y[i:i+10]
    header_content += "    " + ", ".join([f"{val:.6f}f" for val in chunk]) + ",\n"

header_content += "};\n"

# 3. Ghi ra file vào thư mục src
output_file = os.path.join("src", "dummy_audio.h")
try:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(header_content)
    print(f"✅ Đã tạo xong file {output_file}!")
except Exception as e:
    print(f"❌ Lỗi khi ghi file output: {e}")

import os
import argparse

def extract_audio(input_file, output_file):
    try:
        import librosa
    except ImportError:
        print("Error: librosa is not installed. Please install it using: pip install librosa")
        return

    import numpy as np

    print(f"Reading audio file: {input_file} ...")
    # Load audio: resample to 16kHz, convert to mono
    y, sr = librosa.load(input_file, sr=16000, mono=True)
    
    # Cắt hoặc đệm (pad) để lấy chính xác 16000 samples (1 giây)
    target_samples = 16000
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)), 'constant')

    # Chuyển đổi sang định dạng 16-bit PCM (-32768 đến 32767)
    y_int16 = np.int16(y * 32767)

    print(f"Saving dummy_audio array to: {output_file} ...")
    
    # Tạo thư mục chứa file nếu chưa có
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("#ifndef DUMMY_AUDIO_H\n")
        f.write("#define DUMMY_AUDIO_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// Source file: {os.path.basename(input_file)}\n")
        f.write(f"// Sample Rate: 16000 Hz, Samples: {target_samples} (1 second)\n")
        f.write(f"const int16_t dummy_audio[{target_samples}] = {{\n")
        
        # Ghi dữ liệu vào mảng, mỗi dòng 12 phần tử
        for i in range(0, len(y_int16), 12):
            chunk = y_int16[i:i+12]
            f.write("    " + ", ".join(str(val) for val in chunk))
            if i + 12 < len(y_int16):
                f.write(",")
            f.write("\n")
            
        f.write("};\n\n")
        f.write("#endif // DUMMY_AUDIO_H\n")

    print(f"Successfully exported to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from WAV/MP3 to C-array .h for ESP32.")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file")
    
    # Mặc định xuất vào thư mục 'kitchen/src' nằm cùng cấp với script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(base_dir, "kitchen", "src", "dummy_audio.h")
    
    parser.add_argument("--output", type=str, default=default_output, help="Path to output .h file")
    
    args = parser.parse_args()
    
    # Kiểm tra input file
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
    else:
        extract_audio(args.input, args.output)

import numpy as np
from scipy.fftpack import dct

def extract_mfccs(audio, sample_rate, n_mfcc=13, n_mels=40, n_fft=2048, hop_length=512):
    # 1. Pre-emphasis (Lọc tần số cao, giảm các tần số thấp dư thừa)
    pre_emphasis = 0.97 # Hệ số khuếch đại (thường từ 0.95 đến 0.97)
    # y[t] = x[t] - a*x[t-1]. Hàm np.append ghép mẫu đầu tiên giữ nguyên với mảng đã được trừ đi phần trước đó nhân hệ số.
    emphasized_signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # 2. Phân khung (Framing) và làm đệm (Padding)
    frame_length, frame_step = n_fft, hop_length # Gán độ dài khung (2048) và bước trượt (512)
    signal_length = len(emphasized_signal) # Tổng số lượng mẫu âm thanh có trong tín hiệu
    # Tính toán tổng số lượng khung (frames) có thể được tạo ra, làm tròn lên bằng np.ceil
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # Tính tổng chiều dài yêu cầu để chứa đủ số khung đã tính mà không bị khuyết đuôi
    pad_signal_length = num_frames * frame_step + frame_length
    # Thêm các số 0 (Zero-padding) vào đuôi mảng tín hiệu để đạt độ dài pad_signal_length
    pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))

    # Tạo mảng indices (chỉ số) để bốc tách tín hiệu 1D thành ma trận 2D các khung (shape: num_frames x frame_length)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    # Dùng numpy slice để lấy ra các phần tử từ tín hiệu đệm gốc đưa vào ma trận frames
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 3. Áp dụng Cửa sổ Hamming (Windowing)
    # Nhân phần tử theo phần tử (element-wise) với cửa sổ Hamming để triệt tiêu nhiễu rò rỉ phổ ở vành khung
    frames *= np.hamming(frame_length)

    # 4. FFT và Năng lượng phổ (Power Spectrum)
    # Chuyển ma trận từ miền thời gian sang tần số bằng FFT (biến đổi Fourier nhanh cho số thực). Sau đó lấy Magnitude (Trị tuyệt đối)
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    # Lấy Magnitude bình phương rồi chia cho n_fft để ra Năng lượng dải phổ (Power Spectrum)
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    # 5. Khởi tạo Cửa sổ lọc Mel (Mel Filterbank) 
    low_freq_mel = 0 # Tần số Mel thấp nhất
    # Tính tần số cực đại Nyquist (Sample_rate / 2) và chuyển đổi sang thang đo Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    # Lấy đều (n_mels + 2) điểm trên thang đo Mel (vì cần 1 điểm tựa trái, 1 điểm tựa đỉnh, 1 tựa phải)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    # Đảo ngược công thức để chuyển các mốc điểm Mel đó trở lại về đơn vị Hz bình thường
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    # Trả ra số thứ tự các Bin (cột FFT) nằm tương ứng với những vùng điểm tần số Hz
    bin = np.floor((n_fft + 1) * hz_points / sample_rate)

    # Khởi tạo ma trận trống cho 40 bộ lọc chập với phổ kích thước (n_fft / 2 + 1)
    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    # Quét qua điền tỷ trọng (window weight) cho mỗi bộ lọc m
    for m in range(1, n_mels + 1):
        # Tính hình sườn tam giác mép trái bộ lọc (quét dần từ m-1 tới đỉnh m)
        for k in range(int(bin[m-1]), int(bin[m])): fbank[m-1, k] = (k - bin[m-1]) / (bin[m] - bin[m-1])
        # Tính hình sườn tam giác  mép phải bộ lọc (quét từ đỉnh m đổ xuống m+1)
        for k in range(int(bin[m]), int(bin[m+1])): fbank[m-1, k] = (bin[m+1] - k) / (bin[m+1] - bin[m])

    # Tính tích vô hướng chấm Năng lượng phổ với ma trận chập lọc Mel đã chuyển vị (Transpose)
    filter_banks = np.dot(pow_frames, fbank.T)
    # Nếu kết quả lọc bị = 0, thay thế ngay bằng 1 số thập phân cực kỳ nhỏ (VD: Float epsilon) để chống lỗi phép toán Log chia cho 0
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # Chuyển năng lượng từ hệ tuyến tính sang hệ Decibel (dB)
    filter_banks = 20 * np.log10(filter_banks) # dB scale

    # 6. Khai triển Cosine Rời rạc (DCT)
    # Áp dụng hàm lượng giác DCT loại 2, chuẩn hóa trực giao (ortho norm) trên toàn bộ filter_banks. Sau đó tách lấy ra 13 phần tử đầu tiên 
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    
    # 7. Tính trung bình (Averaging over time frames)
    # Tính đại lượng trung bình cộng chạy dọc theo trục tung toàn bộ các khung thời gian
    return np.mean(mfcc, axis=0) # Trả về mảng array MFCC có kích thước chính xác bằng 13

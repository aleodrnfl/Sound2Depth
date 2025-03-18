import librosa
import numpy as np
import cv2

def wav_to_mel(wav_path, output_size=(224, 224), sr=16000):
    y, sr = librosa.load(wav_path, sr=sr) 
    
    # 첫 1000ms만 사용
    duration_ms = 1000
    samples_per_ms = sr / 1000
    y = y[:int(duration_ms * samples_per_ms)]
    
    # STFT 수행 (논문 스펙에 맞춤)
    stft = librosa.stft(y, 
                       n_fft=512,  # window length
                       hop_length=160,  # hop length
                       win_length=512)  # window length
    
    # magnitude spectrogram 계산
    magnitude = np.abs(stft)
    
    # amplitude_to_db로 변환
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    magnitude_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min())
    
    magnitude_resized = cv2.resize(magnitude_norm, output_size, interpolation=cv2.INTER_CUBIC)
    
    return magnitude_resized[np.newaxis, ...]
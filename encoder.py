import cv2
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
from tqdm import tqdm
import os

# ===================== CONFIG =====================
INPUT_AUDIO = "input.mp3" # Giriş ses
OUTPUT_VIDEO = "output_hd.avi" # Sabit FPS video
VIDEO_FPS = 120 # Çok yüksek FPS
FRAME_SIZE = (1920, 1080) # Full HD
DOT_RADIUS = 1 # Nokta boyutu
AMPLIFY = 0.9 # X ekseni genliği
SAMPLES_PER_FRAME = 1000 # 120*1000 = 120 kHz

# ===================== ENCODER =====================
print("[ENCODER] Ses yükleniyor...")
audio = AudioSegment.from_file(INPUT_AUDIO)
audio = audio.set_channels(1).set_frame_rate(VIDEO_FPS * SAMPLES_PER_FRAME) # 120kHz
temp_wav = "temp_hd.wav"
audio.export(temp_wav, format="wav")

rate, samples = wav.read(temp_wav)
os.remove(temp_wav)

# Normalize
samples = samples.astype(np.float32)
samples /= np.max(np.abs(samples))

# Pad to multiple of SAMPLES_PER_FRAME
pad_len = (SAMPLES_PER_FRAME - len(samples) % SAMPLES_PER_FRAME) % SAMPLES_PER_FRAME
samples = np.pad(samples, (0, pad_len), mode='constant')

# Frame sayısı
total_frames = len(samples) // SAMPLES_PER_FRAME
print(f"[ENCODER] Frame: {total_frames}, Sample: {len(samples)}")

fourcc = cv2.VideoWriter_fourcc(*'XVID') # AVI için sabit FPS
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, VIDEO_FPS, FRAME_SIZE)

height, width = FRAME_SIZE[1], FRAME_SIZE[0]
center_y = height // 2

for i in tqdm(range(total_frames), desc="[ENCODER] Frame yazılıyor"):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    chunk = samples[i*SAMPLES_PER_FRAME:(i+1)*SAMPLES_PER_FRAME]

    for j, x_val in enumerate(chunk):
        x_pos = int((x_val * (width/2) * AMPLIFY) + (width/2))
        y_pos = int(center_y + (j - len(chunk)/2) * (height/len(chunk)))
        frame[y_pos, x_pos] = (255, 255, 255)

    video_writer.write(frame)

video_writer.release()
print(f"[ENCODER] Tamamlandı: {OUTPUT_VIDEO}")
print(f"[ENCODER] Input süresi ≈ {len(samples)/rate:.2f} saniye")



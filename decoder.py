import cv2
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

# ===================== CONFIG =====================
INPUT_VIDEO = "output_hd.avi"
DECODED_WAV = "decoded_hd.wav"
VIDEO_FPS = 120
FRAME_SIZE = (1920, 1080)
AMPLIFY = 0.9
SAMPLES_PER_FRAME = 1000 # Encoder ile birebir

# ===================== DECODER =====================
print("[DECODER] Video okunuyor...")
cap = cv2.VideoCapture(INPUT_VIDEO)
decoded_samples = []

width, height = FRAME_SIZE
half_width = width / 2

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[DECODER] Frame sayısı: {frame_count}")

for _ in tqdm(range(frame_count), desc="[DECODER] Frame okunuyor"):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))

    frame_samples = []

    if coords.size > 0:
        coords = coords[coords[:,0].argsort()] # Y eksenine göre sırala
        ys = coords[:,0]
        xs = coords[:,1]

        unique_y = np.unique(ys)
        for uy in unique_y[:SAMPLES_PER_FRAME]: # Fazla nokta olursa kırp
            x_vals = xs[ys == uy]
            mean_x = np.mean(x_vals)
            normalized_x = (mean_x - half_width) / half_width / AMPLIFY
            frame_samples.append(normalized_x)

    while len(frame_samples) < SAMPLES_PER_FRAME:
        frame_samples.append(0.0)

    decoded_samples.extend(frame_samples)

cap.release()

decoded_audio = np.array(decoded_samples)
sample_rate = VIDEO_FPS * SAMPLES_PER_FRAME # 120kHz
decoded_audio = (decoded_audio * 32767).astype(np.int16)

wav.write(DECODED_WAV, sample_rate, decoded_audio)
print(f"[DECODER] Tamamlandı: {DECODED_WAV}")
print(f"[DECODER] Output süresi ≈ {len(decoded_samples)/sample_rate:.2f} saniye")
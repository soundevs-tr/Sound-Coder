import tkinter as tk
from tkinter import filedialog, ttk
import os
from threading import Thread
import cv2
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
from tqdm import tqdm

class SoundCoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Coder")
        self.root.geometry("500x300")
        
        # Main title
        tk.Label(root, text="Sound Coder", font=("Arial", 24, "bold")).pack(pady=20)
        
        # Encoder section
        encoder_frame = tk.Frame(root)
        encoder_frame.pack(pady=10)
        tk.Label(encoder_frame, text="Encoder", font=("Arial", 14)).pack()
        self.encoder_button = tk.Button(encoder_frame, text="Encode MP3 to AVI", command=self.start_encoder)
        self.encoder_button.pack(pady=5)
        self.encoder_progress = ttk.Progressbar(encoder_frame, orient="horizontal", length=300, mode="determinate")
        self.encoder_progress.pack()
        
        # Decoder section
        decoder_frame = tk.Frame(root)
        decoder_frame.pack(pady=10)
        tk.Label(decoder_frame, text="Decoder", font=("Arial", 14)).pack()
        self.decoder_button = tk.Button(decoder_frame, text="Decode AVI to WAV", command=self.start_decoder)
        self.decoder_button.pack(pady=5)
        self.decoder_progress = ttk.Progressbar(decoder_frame, orient="horizontal", length=300, mode="determinate")
        self.decoder_progress.pack()
        
        # Status label
        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack(pady=10)
        
        # Initialize variables
        self.encoding = False
        self.decoding = False
    
    def start_encoder(self):
        if self.encoding:
            return
        file_path = filedialog.askopenfilename(title="Select MP3 file", filetypes=[("MP3 files", "*.mp3")])
        if file_path:
            self.encoding = True
            self.encoder_button.config(state="disabled")
            self.status_label.config(text="Encoding in progress...", fg="blue")
            Thread(target=self.encode_audio, args=(file_path,)).start()
    
    def start_decoder(self):
        if self.decoding:
            return
        file_path = filedialog.askopenfilename(title="Select AVI file", filetypes=[("AVI files", "*.avi")])
        if file_path:
            self.decoding = True
            self.decoder_button.config(state="disabled")
            self.status_label.config(text="Decoding in progress...", fg="blue")
            Thread(target=self.decode_video, args=(file_path,)).start()
    
    def encode_audio(self, input_file):
        try:
            # Configuration
            output_video = os.path.join(os.path.expanduser("~"), "output_hd.avi")
            video_fps = 120
            frame_size = (1920, 1080)
            dot_radius = 1
            amplify = 0.9
            samples_per_frame = 1000
            
            # Load and convert audio
            self.update_status("Loading audio...")
            audio = AudioSegment.from_file(input_file)
            audio = audio.set_channels(1).set_frame_rate(video_fps * samples_per_frame)  # 120kHz
            temp_wav = "temp_hd.wav"
            audio.export(temp_wav, format="wav")
            
            rate, samples = wav.read(temp_wav)
            os.remove(temp_wav)
            
            # Normalize
            samples = samples.astype(np.float32)
            samples /= np.max(np.abs(samples))
            
            # Pad to multiple of samples_per_frame
            pad_len = (samples_per_frame - len(samples) % samples_per_frame) % samples_per_frame
            samples = np.pad(samples, (0, pad_len), mode='constant')
            
            # Frame count
            total_frames = len(samples) // samples_per_frame
            self.update_status(f"Encoding {total_frames} frames...")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_video, fourcc, video_fps, frame_size)
            
            height, width = frame_size[1], frame_size[0]
            center_y = height // 2
            
            # Encoding loop
            for i in range(total_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                chunk = samples[i*samples_per_frame:(i+1)*samples_per_frame]
                
                for j, x_val in enumerate(chunk):
                    x_pos = int((x_val * (width/2) * amplify) + (width/2))
                    y_pos = int(center_y + (j - len(chunk)/2) * (height/len(chunk)))
                    frame[y_pos, x_pos] = (255, 255, 255)
                
                video_writer.write(frame)
                progress = (i + 1) / total_frames * 100
                self.update_progress(self.encoder_progress, progress)
            
            video_writer.release()
            self.update_status(f"Encoding complete! Saved to {output_video}", "green")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
        finally:
            self.encoding = False
            self.encoder_button.config(state="normal")
    
    def decode_video(self, input_file):
        try:
            # Configuration
            output_wav = os.path.join(os.path.expanduser("~"), "decoded_hd.wav")
            video_fps = 120
            frame_size = (1920, 1080)
            amplify = 0.9
            samples_per_frame = 1000
            
            self.update_status("Reading video...")
            cap = cv2.VideoCapture(input_file)
            decoded_samples = []
            
            width, height = frame_size
            half_width = width / 2
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.update_status(f"Decoding {frame_count} frames...")
            
            # Decoding loop
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                coords = np.column_stack(np.where(gray > 0))
                
                frame_samples = []
                
                if coords.size > 0:
                    coords = coords[coords[:,0].argsort()]  # Sort by Y axis
                    ys = coords[:,0]
                    xs = coords[:,1]
                    
                    unique_y = np.unique(ys)
                    for uy in unique_y[:samples_per_frame]:  # Trim if too many points
                        x_vals = xs[ys == uy]
                        mean_x = np.mean(x_vals)
                        normalized_x = (mean_x - half_width) / half_width / amplify
                        frame_samples.append(normalized_x)
                
                while len(frame_samples) < samples_per_frame:
                    frame_samples.append(0.0)
                
                decoded_samples.extend(frame_samples)
                progress = (i + 1) / frame_count * 100
                self.update_progress(self.decoder_progress, progress)
            
            cap.release()
            
            # Save decoded audio
            decoded_audio = np.array(decoded_samples)
            sample_rate = video_fps * samples_per_frame  # 120kHz
            decoded_audio = (decoded_audio * 32767).astype(np.int16)
            
            wav.write(output_wav, sample_rate, decoded_audio)
            self.update_status(f"Decoding complete! Saved to {output_wav}", "green")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
        finally:
            self.decoding = False
            self.decoder_button.config(state="normal")
    
    def update_status(self, message, color="blue"):
        self.root.after(0, lambda: self.status_label.config(text=message, fg=color))
    
    def update_progress(self, progressbar, value):
        self.root.after(0, lambda: progressbar.config(value=value))

if __name__ == "__main__":
    root = tk.Tk()
    app = SoundCoderApp(root)
    root.mainloop()
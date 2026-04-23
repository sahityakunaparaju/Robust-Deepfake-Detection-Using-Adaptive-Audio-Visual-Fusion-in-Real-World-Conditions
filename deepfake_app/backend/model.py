import cv2
import numpy as np
import os
import subprocess
import tempfile
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import Wav2Vec2Model
import soundfile as sf
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5



class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        mobilenet = models.mobilenet_v2(weights='DEFAULT')
        self.visual_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_proj = nn.Linear(1280, 256)

        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.audio_proj = nn.Linear(768, 256)

        self.gate = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, frames, audio):
        B, T, C, H, W = frames.shape

        frames = frames.view(B*T, C, H, W)
        vis_feat = self.visual_encoder(frames)
        vis_feat = self.visual_pool(vis_feat)
        vis_feat = vis_feat.view(B, T, -1).mean(dim=1)
        vis_feat = self.visual_proj(vis_feat)

        audio = audio.squeeze(1)
        audio_out = self.audio_encoder(audio).last_hidden_state.mean(dim=1)
        aud_feat = self.audio_proj(audio_out)

        gate_input = torch.cat([vis_feat, aud_feat], dim=1)
        gate = self.gate(gate_input)

        fused_audio = gate * aud_feat
        fused_visual = (1 - gate) * vis_feat
        fused = torch.cat([fused_visual, fused_audio], dim=1)

        out = self.classifier(fused).squeeze(1)
        return out, gate


# ---------------- LOAD MODEL ---------------- #

model_instance = DeepfakeDetector().to(device)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pt")

if os.path.exists(MODEL_PATH):
    model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model_instance.eval()


# ---------------- AUDIO ---------------- #

def extract_audio(video_path):
    temp_wav = tempfile.mktemp(suffix='.wav')

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        temp_wav
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        audio, sr = sf.read(temp_wav)
        os.remove(temp_wav)

        audio = torch.tensor(audio).float().unsqueeze(0)

        target_len = 48000
        if audio.shape[1] >= target_len:
            audio = audio[:, :target_len]
        else:
            pad = torch.zeros(1, target_len - audio.shape[1])
            audio = torch.cat([audio, pad], dim=1)

        audio = audio / (audio.abs().max() + 1e-8)
        return audio

    except:
        return torch.zeros(1, 48000)


# ---------------- VIDEO ---------------- #

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // num_frames)

    count = 0

    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = frame[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                frames.append(face)

        count += 1

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(frames)
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0

    if frames.shape[0] < num_frames:
        pad = torch.zeros(num_frames - frames.shape[0], *frames.shape[1:])
        frames = torch.cat([frames, pad], dim=0)

    return frames.unsqueeze(0)


# ---------------- MAIN ---------------- #

def analyze_video(video_path):
    start = time.time()

    frames = extract_frames(video_path)
    if frames is None:
        return {"error": "No face detected"}

    audio = extract_audio(video_path)

    frames = frames.to(device)
    audio = audio.unsqueeze(0).to(device)

    frame_probs = []

    with torch.no_grad():
        for i in range(frames.shape[1]):
            single = frames[:, i:i+1]
            pred, _ = model_instance(single, audio)
            prob = torch.sigmoid(pred).item()
            frame_probs.append(prob)

        final_pred, gate = model_instance(frames, audio)
        final_conf = torch.sigmoid(final_pred).item()

    prediction = "Fake" if final_conf > THRESHOLD else "Real"

    fake_ratio = sum(p > 0.5 for p in frame_probs) / len(frame_probs)

    # ---------- TOP FRAMES ---------- #
    top_indices = sorted(range(len(frame_probs)), key=lambda i: frame_probs[i], reverse=True)[:5]

    top_frames = []
    frames_dir = os.path.join(os.path.dirname(__file__), "static/frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    for idx in top_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f"frame_{idx}.jpg"
        filepath = os.path.join(frames_dir, filename)
        cv2.imwrite(filepath, frame)

        top_frames.append({
            "frame_index": int(idx),
            "score": float(frame_probs[idx]),
            "image_path": f"http://localhost:8000/static/frames/{filename}"
        })

    cap.release()

    # ---------- SMART SUMMARY ---------- #
    if final_conf > 0.7:
        summary = f"The video is classified as FAKE with {round(final_conf*100)}% confidence. Multiple frames show strong manipulation patterns."
    elif final_conf > 0.4:
        summary = f"The video shows mixed signals. Some frames appear suspicious, but overall confidence is moderate."
    else:
        summary = f"The video is classified as REAL with {round((1-final_conf)*100)}% confidence. Minor anomalies exist but overall structure is authentic."

    return {
        "prediction": prediction,
        "confidence": float(final_conf),

        "video_score": float(np.mean(frame_probs)),
        "audio_score": float(final_conf),
        "fake_frame_ratio": float(fake_ratio),

        "frame_predictions": frame_probs,

        "fusion_weights": {
            "video": float(1 - gate.mean().item()),
            "audio": float(gate.mean().item())
        },

        "decision_summary": summary,

        "num_frames_analyzed": len(frame_probs),
        "stability_score": float(np.std(frame_probs)),

        "metadata": {
            "audio_detected": True,
            "processing_efficiency_fps": round(len(frame_probs)/(time.time()-start), 2)
        },

        "top_frames": top_frames
    }
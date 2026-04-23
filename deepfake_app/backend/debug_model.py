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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_instance = DeepfakeDetector().to(device)
model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
if os.path.exists(model_path):
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
model_instance.eval()

def extract_audio_subprocess(video_path):
    temp_wav = tempfile.mktemp(suffix='.wav')
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        temp_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(temp_wav):
            audio_data, sr = sf.read(temp_wav)
            os.remove(temp_wav)
            if len(audio_data) > 0:
                audio_tensor = torch.tensor(audio_data).float().unsqueeze(0)
                target_len = 48000
                if audio_tensor.shape[1] >= target_len:
                    audio_tensor = audio_tensor[:, :target_len]
                else:
                    pad = torch.zeros(1, target_len - audio_tensor.shape[1])
                    audio_tensor = torch.cat([audio_tensor, pad], dim=1)
                audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)
                return audio_tensor
    except Exception:
        pass
    return None

def extract_frames_tensor(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    raw_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    if total_frames > 0:
        step = max(1, total_frames // num_frames)
        count = 0
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                face_detected = len(faces) > 0
                frames.append((frame_resized, face_detected))
                raw_frames.append(frame) 
            count += 1
    cap.release()
    
    faces_found = [f[1] for f in frames]
    frames_only = [f[0] for f in frames]
    
    if not frames_only:
        return None, 0, False, []
        
    frames_tensor = torch.tensor(np.array(frames_only)).permute(0, 3, 1, 2)
    frames_tensor = frames_tensor.float() / 255.0
    
    if frames_tensor.shape[0] < num_frames:
        pad_t = torch.zeros(num_frames - frames_tensor.shape[0], *frames_tensor.shape[1:])
        frames_tensor = torch.cat([frames_tensor, pad_t], dim=0)
        
        while len(raw_frames) < num_frames:
            raw_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    else:
        frames_tensor = frames_tensor[:num_frames]
        raw_frames = raw_frames[:num_frames]
        
    return frames_tensor.unsqueeze(0), duration, any(faces_found), raw_frames

def analyze_video(video_path):
    target_frames = 16
    start_cpu_time = time.time()
    
    frames_tensor, duration, face_detected, raw_frames = extract_frames_tensor(video_path, num_frames=target_frames)
    
    if frames_tensor is None:
        return {"error": "Could not extract frames. The video file is empty or corrupted."}
        
    if not face_detected:
        return {"error": "No human face detected. The system requires facial data to run proper analysis."}
        
    audio_tensor = extract_audio_subprocess(video_path)
    audio_detected = audio_tensor is not None
    
    if not audio_detected:
        audio_tensor = torch.zeros(1, 48000) + 1e-6
        
    frames_tensor = frames_tensor.to(device)
    audio_tensor = audio_tensor.unsqueeze(0).to(device) 
    
    with torch.no_grad():
        preds, gate = model_instance(frames_tensor, audio_tensor)
        conf = torch.sigmoid(preds).item()
        audio_weight = gate.mean().item()
        video_weight = 1.0 - audio_weight

    frame_predictions_scores = []
    for i in range(target_frames):
        single_frame = frames_tensor[:, i:i+1, :, :, :]
        with torch.no_grad():
            f_preds, _ = model_instance(single_frame, audio_tensor)
            f_prob = torch.sigmoid(f_preds).item()
            frame_predictions_scores.append(float(round(f_prob, 4)))

    f_score_pairs = [(idx, score) for idx, score in enumerate(frame_predictions_scores)]
    f_score_pairs_sorted = sorted(f_score_pairs, key=lambda x: x[1], reverse=True)
    
    timestamp = int(time.time() * 1000)
    static_frames_dir = os.path.join(os.path.dirname(__file__), "static", "frames")
    os.makedirs(static_frames_dir, exist_ok=True)
    
    top_frames_data = []
    for rank in range(min(5, len(f_score_pairs_sorted))):
        f_idx, score = f_score_pairs_sorted[rank]
        img_filename = f"frame_{timestamp}_{f_idx}.jpg"
        img_path = os.path.join(static_frames_dir, img_filename)
        cv2.imwrite(img_path, raw_frames[f_idx])
        
        top_frames_data.append({
            "frame_index": f_idx + 1,
            "score": float(round(score, 4)),
            "image_path": f"/static/frames/{img_filename}",
            "absolute_path": img_path
        })
        
    prediction = "Fake" if conf > 0.50 else "Real"
    
    video_score = conf if not audio_detected else (conf * video_weight * 2)
    video_score = min(video_score, 1.0)
    audio_score = conf if audio_detected else 0.0
    
    fake_frame_ratio = sum(1 for p in frame_predictions_scores if p > 0.5) / len(frame_predictions_scores)
    variance_score = float(np.var(frame_predictions_scores)) if len(frame_predictions_scores) > 0 else 0.0
    stability = max(0.0, float(round(1.0 - variance_score, 4)))
    
    vis_pct = int(video_weight * 100)
    aud_pct = 100 - vis_pct if audio_detected else 0
    conf_pct = int(conf * 100)
    
    if prediction == "Fake":
        desc = f"The video is classified as FAKE with {conf_pct}% confidence. "
        desc += f"Key anomalies were detected in frames {', '.join([str(f['frame_index']) for f in top_frames_data[:3]])}. "
        desc += f"Visual inconsistencies contributed {vis_pct}% of the decision"
        if audio_detected:
             desc += f", while audio anomalies contributed {aud_pct}%."
        else:
             desc += "."
    else:
        desc = f"The video is classified as REAL indicating high authenticity. "
        desc += f"The stability score of {stability} indicates consistent authentic patterns across the sequence."
        
    efficiency = float(round(target_frames / max(1e-5, (time.time() - start_cpu_time)), 1))
    
    return {
        "prediction": prediction,
        "confidence": float(round(conf, 4)),
        "video_score": float(round(video_score, 4)),
        "audio_score": float(round(audio_score, 4)),
        "num_frames_analyzed": target_frames,
        "fake_frame_ratio": float(round(fake_frame_ratio, 4)),
        "frame_predictions": frame_predictions_scores,
        "top_frames": top_frames_data,
        "fusion_weights": {
            "video": float(round(video_weight, 4)),
            "audio": float(round(audio_weight, 4))
        },
        "stability_score": stability,
        "decision_summary": desc,
        "metadata": {
            "duration_seconds": round(duration, 2),
            "audio_detected": audio_detected,
            "detection_stability_score": stability,
            "processing_efficiency_fps": efficiency
        }
    }

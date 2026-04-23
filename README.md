# Explainable Deepfake Analysis

An end-to-end multimodal deepfake detection system that combines visual and audio evidence, exposes modality-level fusion weights, and provides frame-level interpretability through an interactive dashboard and downloadable PDF reports.

## 1) Project Overview

This repository implements a practical deepfake analysis stack with three core goals:

- **Detection**: classify an input video as `Fake` or `Real`.
- **Interpretability**: expose why the decision was made (frame-level scores, top suspicious frames, modality contributions).
- **Operational usability**: provide a browser dashboard and report export API for reproducible analysis workflows.

The backend is built with FastAPI, the model pipeline uses PyTorch + torchvision + Wav2Vec2, and the frontend is a static HTML/CSS/JS interface with Chart.js visualizations.

## 2) Dashboard Samples

The following screenshots are included under `assets/`:

- Real-classification dashboard view  
  `assets/dashboard-real.png`
- Extended interpretability section with top manipulated frames and advanced metrics  
  `assets/dashboard-details.png`
- Fake-classification dashboard view  
  `assets/dashboard-fake.png`

To display them in markdown:

![Real prediction dashboard](assets/dashboard-real.png)
![Detailed interpretability dashboard](assets/dashboard-details.png)
![Fake prediction dashboard](assets/dashboard-fake.png)

## 3) System Architecture

### 3.1 Backend Service

Located in `backend/main.py`:

- `POST /api/analyze`: accepts `.mp4` / `.avi`, runs model inference, returns structured explainability payload.
- `POST /api/report`: generates a PDF report from returned analysis JSON.
- Static file serving at `/static` for exported top-frame images.

### 3.2 Inference Model

Implemented in `backend/model.py`:

- **Visual encoder**: `MobileNetV2` feature extractor (`torchvision`).
- **Audio encoder**: `Wav2Vec2 base` (`facebook/wav2vec2-base`, frozen during downstream inference).
- **Fusion**: learned scalar gate over concatenated modality embeddings.
- **Classifier**: MLP over fused visual/audio representation.

The gate provides an interpretable estimate of audio vs visual reliance:

- `fusion_weights.audio = mean(gate)`
- `fusion_weights.video = 1 - mean(gate)`

### 3.3 Frontend Interface

Located in `frontend/`:

- Drag-and-drop upload and preview workflow.
- Live analysis progress overlay.
- Results dashboard with:
  - final prediction and confidence gauge,
  - frame-level manipulation timeline,
  - modality scores and fusion weights,
  - top manipulated frame thumbnails,
  - metadata (stability, audio presence, processing FPS),
  - one-click PDF report download.

## 4) Inference and Explainability Pipeline

For each uploaded video:

1. **Frame extraction**: uniformly sample frames from the sequence; detect/crop face regions using Haar cascades; resize to `224x224`.
2. **Audio extraction**: use `ffmpeg` to resample mono waveform (`16 kHz`), normalize amplitude, and pad/truncate to fixed length.
3. **Per-frame scoring**: run single-frame passes to estimate frame-level fake probabilities.
4. **Holistic prediction**: run full sequence pass for final video confidence.
5. **Top-frame mining**: rank frames by manipulation score and export top-5 thumbnails.
6. **Structured output**:
   - `prediction`, `confidence`
   - `video_score`, `audio_score`, `fake_frame_ratio`
   - `frame_predictions`
   - `fusion_weights`
   - `decision_summary`
   - `metadata` and `top_frames`

## 5) Repository Layout

```text
deepfake_app/
├── backend/
│   ├── main.py               # FastAPI service, upload + report endpoints
│   ├── model.py              # Inference model and analysis pipeline
│   ├── debug.py              # Diagnostic audit script
│   ├── debug_model.py        # Alternative debug/inference variant
│   └── requirements.txt
├── frontend/
│   ├── index.html            # Dashboard structure
│   ├── style.css             # UI styling
│   └── script.js             # Upload flow, API calls, chart rendering
├── uploads/
│   └── deepfake_detection.py # Training/experimentation script snapshot
├── assets/                   # Dashboard screenshots
└── README.md
```

## 6) Setup and Execution

## Prerequisites

- Python 3.10+ recommended
- `ffmpeg` available on system PATH
- Optional: CUDA-enabled GPU for faster inference

## Installation

```bash
cd deepfake_app/backend
pip install -r requirements.txt
```

## Start Backend API

```bash
cd deepfake_app/backend
python main.py
```

Server default: `http://localhost:8000`

## Launch Frontend

Open `deepfake_app/frontend/index.html` in a browser (or serve it with any static file server).  
Ensure backend is running on `localhost:8000` as expected by `frontend/script.js`.

## 7) API Contract

### `POST /api/analyze`

Request:

- multipart form-data with file field: `video`

Response (abridged):

```json
{
  "prediction": "Fake",
  "confidence": 0.84,
  "video_score": 0.95,
  "audio_score": 0.84,
  "fake_frame_ratio": 1.0,
  "frame_predictions": [0.82, 0.70, 0.76],
  "fusion_weights": { "video": 0.56, "audio": 0.44 },
  "stability_score": 0.98,
  "decision_summary": "The video is classified as FAKE...",
  "metadata": {
    "audio_detected": true,
    "processing_efficiency_fps": 0.6
  },
  "top_frames": [
    { "frame_index": 15, "score": 0.97, "image_path": "/static/frames/..." }
  ]
}
```

### `POST /api/report`

Request:

- JSON payload from `/api/analyze` response.

Response:

- downloadable PDF (`analysis_report.pdf`).

## 8) Research Context and Design Rationale

This implementation follows a multimodal robustness hypothesis: deepfake traces may surface differently across modalities, and modality reliability can vary by sample. The gating mechanism is intended to adaptively balance visual and audio evidence rather than rely on static fusion.

Interpretability is operationalized in three layers:

- **Global**: final class + confidence.
- **Modal**: learned fusion weights (visual vs audio contribution).
- **Temporal**: per-frame fake likelihood and ranked suspicious frames.

This aligns with practical forensic workflows where analysts need traceable supporting evidence, not only scalar predictions.

## 9) Known Limitations and Technical Notes

- `best_model.pt` is required for meaningful inference; otherwise weights remain randomly initialized.
- Haar cascade face detection is lightweight but can fail in profile views, low light, blur, or occlusion.
- Audio extraction depends on `ffmpeg`; missing binary leads to silent fallback behavior in pipeline.
- Fixed-size temporal/audio windows simplify deployment but may underrepresent long-form manipulations.
- `uploads/deepfake_detection.py` indicates training-time experimentation that may not exactly match production inference defaults (for example, frame count choices), so strict train/inference parity should be verified before benchmarking claims.
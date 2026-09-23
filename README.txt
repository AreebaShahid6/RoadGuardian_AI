# 🚗 RoadGuardian AI

### AI-Powered Road Hazard & Pothole Detection

RoadGuardian AI is a **computer vision-based road monitoring system** designed to detect potholes and other road hazards from video footage using trained **YOLO object detection models**.

The system processes road videos locally, identifies detected hazards, assigns tracking IDs, and generates an annotated output video for analysis and demonstration.

---

## 🎯 Key Features

* 🕳️ **Pothole Detection** using a trained YOLO model
* ⚠️ **Road Hazard Detection** using an optional hazard model
* 🎯 **Object Tracking** using YOLO's built-in ByteTrack tracker
* 🔢 **Stable Detection IDs** for tracked objects
* 🛡️ **Anti-Flicker Memory** to reduce disappearing detection boxes
* 📦 **Bounding Box Smoothing** to reduce box jitter
* 🎥 **Video-Based Inference**
* 💻 **Local Inference** without requiring cloud deployment
* ⚙️ **Configurable Confidence Threshold**
* 📁 Automatic generation of annotated output videos

---

## 🧠 How It Works

```text
        Input Road Video
               │
               ▼
       ┌─────────────────┐
       │   YOLO Models   │
       │                 │
       │ Pothole Model   │
       │ Hazard Model    │
       └────────┬────────┘
                │
                ▼
       Object Detection
                │
                ▼
       ByteTrack Tracking
                │
                ▼
       Anti-Flicker Memory
                │
                ▼
        Box Smoothing
                │
                ▼
      Annotated Output Video
```

---

## 📂 Project Structure

```text
RoadGuardian_AI/
│
├── models/
│   ├── hazard_best.pt
│   └── pothole_best.pt
│
├── video/
│   └── input_video.mp4
│
├── output/
│   └── result.mp4
│
├── detect.py
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** Model weights, input videos, and generated output files can be excluded from GitHub using `.gitignore` because they may be large.

---

## 🛠️ Technologies Used

| Technology             | Purpose                     |
| ---------------------- | --------------------------- |
| 🐍 Python              | Core programming language   |
| 🤖 YOLO                | Object detection            |
| 🎯 ByteTrack           | Object tracking             |
| 👁️ OpenCV             | Video processing            |
| 📦 Ultralytics         | YOLO inference and tracking |
| 💻 Virtual Environment | Dependency management       |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AreebaShahid6/RoadGuardian_AI.git
```

Move into the project directory:

```bash
cd RoadGuardian_AI
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Model Setup

Place your trained model files inside the `models/` directory.

### Pothole Model

```text
models/pothole_best.pt
```

### Optional Hazard Model

```text
models/hazard_best.pt
```

The system automatically detects which models are available.

### Pothole model only

```text
pothole_best.pt
        ↓
Pothole Detection
```

### Both models

```text
pothole_best.pt + hazard_best.pt
              ↓
      Combined Detection
```

The system can run with **only the pothole model**. The hazard model is optional.

---

# 🎥 Input Video

Place your test video inside:

```text
video/input_video.mp4
```

You can also provide a custom video path using the `--video` argument.

---

# 🚀 Running the Application

## Basic Detection

If the models and video use the default names:

```bash
python detect.py
```

The annotated result will be generated in:

```text
output/result.mp4
```

---

## Custom Video and Models

```bash
python detect.py --video video/myclip.mp4 --hazard models/hazard_best.pt --pothole models/pothole_best.pt --output output/result.mp4
```

---

## Pothole-Only Detection

If you only have a pothole model:

```bash
python detect.py --pothole models/pothole_best.pt --video video/input_video.mp4
```

---

# 🎯 Detection Confidence

The default confidence threshold is:

```text
0.15
```

You can change it using:

```bash
python detect.py --conf 0.25
```

### Confidence Guide

| Confidence | Behavior                             |
| ---------- | ------------------------------------ |
| `0.15`     | Detects more faint/uncertain objects |
| `0.20`     | Balanced detection                   |
| `0.25`     | Fewer false positives                |
| `0.30+`    | More strict detection                |

A lower threshold may detect more potholes but can also increase false positives.

---

# 🎯 Object Tracking with ByteTrack

RoadGuardian AI uses **ByteTrack** to improve detection consistency across video frames.

Without tracking:

```text
Frame 1 → Pothole detected
Frame 2 → Pothole detected
Frame 3 → New detection
Frame 4 → Detection disappears
```

With tracking:

```text
Frame 1 → P1
Frame 2 → P1
Frame 3 → P1
Frame 4 → P1
```

This provides more consistent object identities throughout the video.

### Detection ID Format

```text
P1 → Pothole
P2 → Pothole
H1 → Hazard
H2 → Hazard
```

`P` represents a pothole and `H` represents a road hazard.

---

# 🛡️ Anti-Flicker Detection

Even with tracking, a detection may temporarily disappear when the model's confidence drops.

RoadGuardian AI includes a **hold buffer** to keep the last known bounding box visible for a configurable number of frames.

### Example

```bash
python detect.py --hold 15
```

Default:

```text
--hold 10
```

### Hold Settings

| Hold  | Behavior                                  |
| ----- | ----------------------------------------- |
| `5`   | More responsive, potentially more flicker |
| `10`  | Balanced                                  |
| `15`  | Smoother, less flickering                 |
| `20+` | Longer persistence but may appear delayed |

A higher value keeps a detection visible longer, but the box may temporarily remain at its previous position.

---

# 📐 Bounding Box Smoothing

Raw object detection can cause bounding boxes to move slightly between frames.

RoadGuardian AI provides a `--smooth` parameter to reduce this jitter.

### Example

```bash
python detect.py --smooth 0.2
```

### Recommended Values

| Smooth | Behavior                       |
| ------ | ------------------------------ |
| `0.2`  | Very stable, slower response   |
| `0.3`  | Smooth and stable              |
| `0.4`  | Balanced default               |
| `0.6`  | Faster response, more movement |

For a demonstration video, a combination such as:

```bash
python detect.py --hold 15 --smooth 0.3
```

can provide smoother-looking detections.

---

# 📊 Detection Pipeline

```text
Video Input
     │
     ▼
Frame Extraction
     │
     ▼
YOLO Object Detection
     │
     ├───────────────┐
     ▼               ▼
Pothole Model    Hazard Model
     │               │
     └───────┬───────┘
             ▼
       ByteTrack
             │
             ▼
     Detection IDs
             │
             ▼
      Anti-Flicker
             │
             ▼
       Box Smoothing
             │
             ▼
    Annotated Video
             │
             ▼
     output/result.mp4
```

---

# 📤 Output

The processed video is automatically saved as:

```text
output/result.mp4
```

The output video contains:

* Bounding boxes
* Detection labels
* Confidence scores
* Tracking IDs
* Pothole detections
* Hazard detections when the hazard model is available

---

# ⚡ Performance

RoadGuardian AI can run locally on a CPU or compatible GPU.

### CPU

CPU inference is supported but can be significantly slower.

For quick testing, use a short video of approximately **10–15 seconds**.

### GPU

A CUDA-compatible GPU can significantly improve inference speed when the required PyTorch/CUDA environment is configured correctly.

---

# 🔧 Troubleshooting

### ❌ Video not found

Check that your video exists:

```text
video/input_video.mp4
```

Or provide the correct path:

```bash
python detect.py --video video/myvideo.mp4
```

---

### ❌ Neither model was found

Make sure your model files are directly inside:

```text
models/
```

For example:

```text
models/pothole_best.pt
models/hazard_best.pt
```

and not:

```text
models/my_folder/pothole_best.pt
```

---

### ❌ Slow processing

If you are running inference on CPU, processing can be slow.

For testing, use a shorter video.

---

### ❌ OpenCV installation problem

Try:

```bash
pip install opencv-python-headless
```

---

# 🔒 Large Files

Large model weights and videos are intentionally excluded from the Git repository.

Examples:

```text
*.pt
*.mp4
*.avi
*.mov
*.mkv
```

This keeps the GitHub repository lightweight and easier to clone.

---

# 👩‍💻 Author

**Areeba Shahid**

Computer Science Graduate | Computer Vision Engineer | Machine Learning Enthusiast

### Connect with me

* 💼 LinkedIn: [Areeba Shahid](https://www.linkedin.com/in/areeba-shahid-1b53b231/)
* 🐙 GitHub: [AreebaShahid6](https://github.com/AreebaShahid6)

---

## ⭐ Project

If you find **RoadGuardian AI** useful or interesting, consider giving the repository a ⭐ on GitHub!

**RoadGuardian AI — Making Roads Safer with Computer Vision.** 🚗🛣️


RoadGuardian AI — Local Inference Setup
========================================

FOLDER LAYOUT (already set up for you — just drop your files in)
--------------------------------------------------------------
RoadGuardian_AI/
├── models/
│   ├── hazard_best.pt      <- put your trained hazard model here (optional)
│   └── pothole_best.pt     <- put your trained pothole model here
├── video/
│   └── input_video.mp4     <- put your test video here (rename to this, or use --video)
├── output/                 <- annotated result video appears here automatically
├── detect.py                <- the inference script
├── requirements.txt
└── README.txt (this file)

You only need to place whichever model(s) you actually trained. The
script auto-detects which ones exist — if only pothole_best.pt is
there, it runs pothole-only. If both are there, it draws both
models' detections onto the same output video (hazard boxes in
orange, pothole boxes in red).


SETUP — run these once
-----------------------
1. Open a terminal and cd into this folder:
     cd path/to/RoadGuardian_AI

2. (Recommended) create a virtual environment:
     python -m venv venv
     venv\Scripts\activate        (Windows)
     source venv/bin/activate     (Mac/Linux)

3. Install dependencies:
     pip install -r requirements.txt


RUNNING DETECTION
------------------
Simplest — if your files match the default names above:
     python detect.py

Custom filenames/paths:
     python detect.py --video video/myclip.mp4 --hazard models/hazard_best.pt --pothole models/pothole_best.pt --output output/result.mp4

Only pothole model:
     python detect.py --pothole models/pothole_best.pt --video video/myclip.mp4

Adjust detection confidence threshold (default 0.15 — lower = catches more/fainter potholes, but risks more false positives; raise it if you start seeing junk detections):
     python detect.py --conf 0.25

WHAT CHANGED — TRACKING
------------------------
This version uses YOLO's built-in ByteTrack tracker instead of running
each frame as a brand-new, independent detection. This fixes three
things at once:
  - Boxes no longer flicker on/off frame to frame — the tracker keeps
    an object "alive" for a few frames even if the model briefly misses it.
  - Each detected object gets a stable ID (e.g. P3, P7) that stays the
    same as the camera moves, instead of looking like a new detection
    every frame. "H" prefix = hazard model, "P" prefix = pothole model.
  - Lowering the confidence threshold to 0.15 means fainter/blurrier
    potholes that were being filtered out now get picked up too.

If you now see too many false positives (boxes on things that aren't
potholes), raise --conf back up toward 0.25-0.3 — it's a trade-off
between catching more real potholes and avoiding junk detections.


STILL FLICKERING? — ANTI-FLICKER MEMORY
-----------------------------------------
Tracking alone gives objects a stable ID, but by default a box still
only appears on frames where the model actually redetects it — if
confidence dips for a frame or two, the box can blink off.

This version adds a --hold buffer: when a tracked object briefly stops
being detected, its last known box keeps being drawn (with a thinner
outline to show it's "held") for up to --hold frames before it's
removed. Default is 10 frames (~0.4s at 25fps).

     python detect.py --hold 15      (smoother, less flicker, box freezes a bit longer during gaps)
     python detect.py --hold 5       (more responsive, closer to raw detection, may flicker more)

Note: during a held frame, the box stays frozen at its last position —
it doesn't move with the camera until the model redetects it. That's
a deliberate trade-off: a briefly-frozen box beats a flickering one,
but very high --hold values can look like the box is "lagging behind"
a fast-moving pothole. 10-15 is a good starting range.


BOX WOBBLING / JITTERING EVEN WHILE DETECTED? — SMOOTHING
------------------------------------------------------------
Separate from flicker (box disappearing), you might notice the box
edges wobble slightly frame to frame even while a pothole is being
detected continuously. This is normal raw model output noise — the
--smooth setting fixes it by blending each new box with its recent
position instead of jumping straight to the new coordinates.

     python detect.py --smooth 0.2     (very steady, but reacts slower to real movement)
     python detect.py --smooth 0.6     (snappier, but more wobble)

Default is 0.4 — a balance between steady and responsive. If your
supervisor wants maximally stable boxes for a demo video, try 0.2-0.3
combined with --hold 15.


OUTPUT
------
The annotated video is written to output/result.mp4 (or wherever
you pointed --output). Open it with any video player to see the
detections drawn frame by frame.


TROUBLESHOOTING
----------------
- "Video not found" -> check the video is actually inside video/ and the filename matches what you passed to --video.
- "Neither model was found" -> make sure hazard_best.pt / pothole_best.pt are directly inside models/, not in a subfolder.
- Very slow processing -> this runs on CPU by default if you don't have a CUDA GPU locally. That's expected — inference on CPU is much slower than the GPU training on Kaggle. For a quick check, trim your test video to 10-15 seconds first.
- If pip install fails on opencv-python, try: pip install opencv-python-headless instead.

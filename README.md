# VisionDash 👁️

**A computer vision project that aims to turn eye blinks into Morse code and eventually into readable text.**

VisionDash is an experimental Python project using webcam-based eye tracking and computer vision.

The initial goal is to detect facial landmarks, track eye movements, identify eye states (open or closed), and eventually recognize blink patterns.

## Current Progress

### Task 01: Webcam-Based Eye Detection

- [x] Open webcam using OpenCV.
- [x] Detect face using MediaPipe Face Landmarker.
- [x] Extract left and right eye landmarks.
- [x] Calculate Eye Aspect Ratio (EAR).
- [x] Classify eyes as open or closed.

### Upcoming Tasks

- [ ] Task 02: Detect and count complete blinks.
- [ ] Task 03: Measure blink duration.
- [ ] Task 04: Convert short and long blinks into Morse code.
- [ ] Task 05: Translate Morse code into readable text.

## Tech Stack

- **Python** – Main programming language.
- **OpenCV** – Webcam access and image processing.
- **MediaPipe** – Face and eye landmark detection.
- **NumPy** – Numerical operations.

## Project Structure

```text
VisionDash/
│
├── models/
│   └── face_landmarker.task
│
├── src/
│   └── main.py
│
├── requirements.txt
└── README.md
```

The MediaPipe model is downloaded automatically when the program runs for the first time, provided it is not already present.

## Installation and Setup

Follow these instructions to run VisionDash on your machine.

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate into the project:

```bash
cd VisionDash
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have the requirements file yet, you can install the packages manually:

```bash
pip install opencv-python mediapipe numpy
```

### 4. Run the application

```bash
python src/main.py
```

On the first run, VisionDash downloads the MediaPipe Face Landmarker model automatically.

Make sure your webcam is connected and accessible.

Press **Q** to close the application.

## How Eye Detection Works

VisionDash uses MediaPipe Face Landmarker to identify facial landmarks.

It extracts six landmark points around each eye and calculates the Eye Aspect Ratio (EAR).

The EAR formula:

\[
EAR = \frac{||P_2-P_6||+||P_3-P_5||}{2||P_1-P_4||}
\]

The EAR value helps determine whether an eye is open or closed.

### Current Detection Threshold

The initial threshold is:

```python
EAR_THRESHOLD = 0.22
```

Initial testing produced these approximate values:

| Eye State | EAR |
|---|---:|
| Open | 0.28–0.32 |
| Closed | 0.03–0.08 |

These measurements are specific to the initial testing setup. The threshold may need calibration for other users and camera conditions.

## Requirements

- Python with a compatible MediaPipe version.
- Webcam.
- Windows, Linux or macOS.
- Internet connection for the initial model download.

## Project Status

**Currently under development.**

VisionDash is an independent learning project focused on computer vision, eye tracking and Morse code communication.

The current version only implements webcam-based eye detection and open/closed eye classification. Blink recognition and Morse translation are still under development.

---

*Built as an independent project, one milestone at a time.*
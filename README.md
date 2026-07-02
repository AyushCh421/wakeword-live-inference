# Wake Word Live Inference

Real-time wake-word detection using a lightweight DS-CNN model optimized for embedded deployment.

---

## Features

- Live microphone recording
- 16 kHz audio processing
- Log-Mel Spectrogram feature extraction
- Real-time wake-word detection
- Confidence score output
- Automatic inference logging
- Lightweight DS-CNN model

---

## Repository Structure

```
model/
inference/
recordings/
reports/
```

---

## Installation

```bash
git clone "https://github.com/AyushCh421/wakeword-live-inference"

cd wakeword-live-inference

pip install -r requirements.txt
```

---

## Run

```bash
python inference/live_inference.py
```

---

## Output

```
Wake Probability

Non-Wake Probability

Detection Status
```

Inference logs are automatically stored in:

```
reports/live_inference_results.csv
```

---

## Model

Architecture:

- DS-CNN
- 15,874 Parameters
- Optimized for TensorFlow Lite
- Target Deployment: ESP32

---

## Current Performance

Validation Accuracy : 99.75%

Precision : 99.68%

Recall : 100%

F1 Score : 99.84%

False Accept Rate : 1.12%

False Reject Rate : 0%

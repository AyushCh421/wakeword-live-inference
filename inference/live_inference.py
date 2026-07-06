import sounddevice as sd
import soundfile as sf
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path

from features import preprocess

# -------------------------------
# Configuration
# -------------------------------

SAMPLE_RATE = 16000
DURATION = 1.5
THRESHOLD = 0.85

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings" / "live_tests"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

RESULTS_FILE = REPORTS_DIR / "live_inference_results.csv"

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sample_no = 1

results = []


# Load Model


print("Loading Model...")


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "final_model_hard_negative.keras"
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded!")

print("=" * 50)
print("Wake Word Listener Started")
print("Press Ctrl+C to Exit")
print("=" * 50)

# -------------------------------

while True:

    input("\nPress ENTER to record (Ctrl+C to Exit)...")

    phrase = input(
    "Enter the phrase you are about to speak: "
            ).strip()

    print("\n🎤 Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()
    filename = RECORDINGS_DIR / f"sample_{sample_no:03d}.wav"
    sf.write(
        filename,
        audio,
        SAMPLE_RATE
    )

    print("Processing...")

    features = preprocess(filename)

    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(
        features,
        verbose=0
    )[0]

    wake_prob = float(prediction[1])
    nonwake_prob = float(prediction[0])

    detected = wake_prob >= THRESHOLD

    results.append({
        "Sample": sample_no,
        "Phrase": phrase,
        "Wake Probability": round(wake_prob, 4),
        "Non-Wake Probability": round(nonwake_prob, 4),
        "Detected": detected,
    })

    pd.DataFrame(results).to_csv(
        RESULTS_FILE,
        index=False
    )

    sample_no += 1

    print("\n" + "=" * 50)
    print(f"Wake Probability     : {wake_prob:.4f}")
    print(f"Non-Wake Probability : {nonwake_prob:.4f}")
    print(f"Threshold            : {THRESHOLD:.2f}")

    if detected:
        print("Status               : ✅ WAKE DETECTED")
    else:
        print("Status               : ❌ NOT DETECTED")

    print("=" * 50)

"""
Feature Extraction for Wake Word Detection
ESP32 Compatible Log-Mel Features
"""

import librosa
import numpy as np

from training.config import (
    SAMPLE_RATE,
    NUM_SAMPLES,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
)


def load_audio(audio_path):
    """
    Load audio as mono 16kHz.
    """

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    return audio


def pad_or_trim(audio):
    """
    Ensure every clip is exactly 1 second.
    If longer than 1 second, take the center.
    """

    if len(audio) > NUM_SAMPLES:

        start = (len(audio) - NUM_SAMPLES) // 2
        audio = audio[start:start + NUM_SAMPLES]

    elif len(audio) < NUM_SAMPLES:

        padding = NUM_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding))

    return audio


def extract_log_mel(audio):
    """
    Compute Log-Mel Spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    return log_mel.astype(np.float32)


def normalize(features):
    """
    Normalize to zero mean and unit variance.
    """

    mean = np.mean(features)
    std = np.std(features)

    return (features - mean) / (std + 1e-6)


def preprocess(audio_path):
    """
    Complete preprocessing pipeline.
    """

    audio = load_audio(audio_path)

    audio = pad_or_trim(audio)

    features = extract_log_mel(audio)

    features = normalize(features)

    return features
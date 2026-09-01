"""Audio feature extraction used by the training datasets.

The implementation is intentionally limited to the mel-spectrogram operation
used by LatentSync. It is built from the public Librosa and NumPy APIs.
"""

from pathlib import Path

import librosa
import numpy as np
from omegaconf import OmegaConf


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "audio.yaml"
config = OmegaConf.load(_CONFIG_PATH)


def _apply_preemphasis(samples: np.ndarray) -> np.ndarray:
    """Apply the first-order high-pass filter configured for training audio."""
    samples = np.asarray(samples, dtype=np.float64)
    if not config.audio.preemphasize or samples.size == 0:
        return samples

    coefficient = config.audio.preemphasis
    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - coefficient * samples[:-1]
    return emphasized


def _normalize_decibels(decibels: np.ndarray) -> np.ndarray:
    """Map the configured decibel range to the model's feature range."""
    min_db = config.audio.min_level_db
    max_value = config.audio.max_abs_value
    unit_interval = (decibels - min_db) / -min_db

    if config.audio.symmetric_mels:
        normalized = 2 * max_value * unit_interval - max_value
        if config.audio.allow_clipping_in_normalization:
            normalized = np.clip(normalized, -max_value, max_value)
    else:
        normalized = max_value * unit_interval
        if config.audio.allow_clipping_in_normalization:
            normalized = np.clip(normalized, 0, max_value)

    if not config.audio.allow_clipping_in_normalization:
        if decibels.max() > 0 or decibels.min() < min_db:
            raise ValueError("Mel-spectrogram values fall outside the configured decibel range")

    return normalized


def melspectrogram(samples: np.ndarray) -> np.ndarray:
    """Convert mono audio samples to normalized mel-spectrogram features."""
    if config.audio.use_lws:
        raise NotImplementedError("The LatentSync training configuration requires Librosa STFT")

    emphasized = _apply_preemphasis(samples)
    magnitude = np.abs(
        librosa.stft(
            y=emphasized,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_size,
            win_length=config.audio.win_size,
        )
    )
    mel_basis = librosa.filters.mel(
        sr=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        n_mels=config.audio.num_mels,
        fmin=config.audio.fmin,
        fmax=config.audio.fmax,
    )
    mel_magnitude = mel_basis @ magnitude

    amplitude_floor = 10.0 ** (config.audio.min_level_db / 20.0)
    decibels = 20.0 * np.log10(np.maximum(amplitude_floor, mel_magnitude))
    decibels -= config.audio.ref_level_db

    if config.audio.signal_normalization:
        return _normalize_decibels(decibels)
    return decibels

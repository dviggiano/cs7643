import os
from pathlib import Path
import numpy as np
import librosa
import torch


# Configuration
input_root = Path('data/musdb18hq')
output_root = input_root / 'spectrograms'
subsets = ['train', 'test']

# Batching parameters
SR = 44100
DURATION_S = 10
CLIP_SAMPLES = SR * DURATION_S
STEP_SAMPLES = CLIP_SAMPLES # for now will do no overlap // 2  # 50% overlap

# STFT parameters, changing them moves dataset from 15gb to 60gb
N_FFT = 512
# N_FFT = 1024
HOP_LENGTH = 512
WIN_LENGTH = 512
# WIN_LENGTH = 1024


def safe_load(path: Path):
    """
    Safely load an audio file using librosa, with fallback and error handling.
    Returns (y, sr) or (None, None) if loading fails.
    """
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        return y, sr
    except Exception as e:
        print(f"Warning: failed to load {path} with default backend: {e}")
        try:
            # Try fallback backend
            y, sr = librosa.load(path, sr=SR, mono=True, backend='audioread')
            return y, sr
        except Exception as e2:
            print(f"Error: could not load {path} at all, skipping: {e2}")
            return None, None


def process_audio_file(audio_path: Path, out_dir: Path):
    """
    Load .wav audio, slice into overlapping clips, compute log-magnitude STFT, and save.
    """
    # Load audio safely
    y, sr = safe_load(audio_path)
    if y is None:
        return
    total_samples = len(y)

    # Iterate through clips
    for clip_idx, start in enumerate(range(0, total_samples - CLIP_SAMPLES + 1, STEP_SAMPLES)):
        end = start + CLIP_SAMPLES
        clip_folder = out_dir / f"clip_{clip_idx:03d}"
        clip_folder.mkdir(parents=True, exist_ok=True)

        # STFT & log-magnitude
        clip = y[start:end]
        spec = librosa.stft(
            clip,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            center=False
        )
        log_mag = np.log1p(np.abs(spec))

        # Save mix or stem tensor
        fname = 'mix.pt' if audio_path.name == 'mixture.wav' else f"{audio_path.stem}.pt"
        tensor = torch.from_numpy(log_mag)
        torch.save(tensor, clip_folder / fname)


def main():
    output_root.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        subset_dir = input_root / subset
        subset_out = output_root / subset
        subset_out.mkdir(parents=True, exist_ok=True)
        print(f"Processing subset '{subset}'")

        for track_dir in subset_dir.iterdir():
            if not track_dir.is_dir():
                continue
            print(f"  Track: {track_dir.name}")

            # Create base output folder for this track
            track_out = subset_out / track_dir.name

            # Process mixture
            mixture = track_dir / 'mixture.wav'
            if mixture.exists():
                process_audio_file(mixture, track_out)

            # Process stems
            for stem in track_dir.glob('*.wav'):
                if stem.name == 'mixture.wav':
                    continue
                process_audio_file(stem, track_out)


if __name__ == '__main__':
    main()

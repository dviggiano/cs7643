#!/usr/bin/env python
# evaluate.py

import os
import glob
import yaml
import torch
import numpy as np
import soundfile as sf
from modeling.models import VanillaCNN, SimpleUNet

def si_sdr(ref, est, eps=1e-8):
    """
    Compute scale-invariant SDR (mono) between two 1D numpy arrays.
    """
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    alpha  = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    e_true = alpha * ref
    e_res  = est - e_true
    return 10 * np.log10((np.sum(e_true**2) + eps) / (np.sum(e_res**2) + eps))

def load_model(cfg, device):
    """
    Build the model (VanillaCNN or SimpleUNet) and load the best checkpoint.
    """
    net_cfg    = cfg['network']
    model_type = net_cfg['model']
    args = {
        'in_channels':     net_cfg['in_channels'],
        'base_filters':    net_cfg['base_filters'],
        'output_channels': net_cfg['output_channels'],
    }

    if model_type == "VanillaCNN":
        model = VanillaCNN(**args)
    elif model_type == "SimpleUNet":
        model = SimpleUNet(**args)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    ckpt_path = cfg.get('checkpoint_path', 'checkpoints/best_model.pth')
    ckpt      = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval()

def main():
    # — load config & device —
    cfg    = yaml.safe_load(open('config.yaml'))
    device = torch.device(cfg.get('device', 'cpu'))

    # — STFT params (must match training) —
    N_FFT   = 512
    HOP_LEN = 512
    WIN_LEN = 512
    window  = torch.hann_window(WIN_LEN).to(device)

    # — load model —
    model = load_model(cfg, device)

    # — figure out where the raw WAVs live —
    spec_root = cfg['data']['root_dir']            # e.g. data/musdb18hq/spectrograms
    wav_root  = os.path.dirname(spec_root)         # e.g. data/musdb18hq
    test_root = os.path.join(wav_root, 'test')

    stem_names = ['bass', 'drums', 'other', 'vocals']
    all_scores = []

    for song_dir in sorted(glob.glob(f"{test_root}/*")):
        if not os.path.isdir(song_dir):
            continue

        # 1) load mixture WAV & collapse to mono
        mix_np, sr = sf.read(os.path.join(song_dir, 'mixture.wav'))
        if mix_np.ndim == 2:
            mix_np = mix_np.mean(axis=1)
        mix = torch.tensor(mix_np, dtype=torch.float32, device=device)

        # 2) compute complex STFT → [freq, time]
        spec = torch.stft(
            mix,
            n_fft=N_FFT,
            hop_length=HOP_LEN,
            win_length=WIN_LEN,
            window=window,
            return_complex=True
        )  # shape: [F+1, T]

        # 3) build 4D inputs → [B=1, C=1, F+1, T]
        mag   = spec.abs().unsqueeze(0).unsqueeze(0)
        phase = torch.angle(spec).unsqueeze(0).unsqueeze(0)

        # 4) predict
        with torch.no_grad():
            pred_mag = model(mag)  # [1, 4, F+1, T]

        # 5) ISTFT each channel → mono waveform
        preds = []
        for c in range(pred_mag.size(1)):
            comp = pred_mag[0, c] * torch.exp(1j * phase[0, 0])
            wav_est = torch.istft(
                comp,
                n_fft=N_FFT,
                hop_length=HOP_LEN,
                win_length=WIN_LEN,
                window=window,
                length=len(mix_np)
            )
            preds.append(wav_est.cpu().numpy())
        preds = np.stack(preds, axis=0)  # [4, N]

        # 6) load & mono-ify the ground-truth stems
        trues = []
        for s in stem_names:
            w, _ = sf.read(os.path.join(song_dir, f"{s}.wav"))
            if w.ndim == 2:
                w = w.mean(axis=1)
            # pad/trim to match length
            if w.shape[0] > preds.shape[1]:
                w = w[: preds.shape[1]]
            elif w.shape[0] < preds.shape[1]:
                w = np.pad(w, (0, preds.shape[1] - w.shape[0]))
            trues.append(w)
        trues = np.stack(trues, axis=0)  # [4, N]

        # 7) compute SI-SDR per stem
        scores = [si_sdr(trues[i], preds[i]) for i in range(len(stem_names))]
        all_scores.append(scores)

        scores_str = "  ".join(f"{stem_names[i]}: {scores[i]:6.2f} dB"
                               for i in range(len(stem_names)))
        print(f"{os.path.basename(song_dir)} | {scores_str}")

    # 8) report averages
    all_scores = np.array(all_scores)  # [n_songs, 4]
    avg_scores = all_scores.mean(axis=0)
    print("\n=== Average SI-SDR per stem ===")
    for i, stem in enumerate(stem_names):
        print(f"  {stem:6s}: {avg_scores[i]:6.2f} dB")
    print(f"  Overall: {avg_scores.mean():6.2f} dB")

if __name__ == "__main__":
    main()

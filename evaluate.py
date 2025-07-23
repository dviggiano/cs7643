#!/usr/bin/env python
# evaluate_vocals.py

import os
import glob
import yaml
import torch
import numpy as np
import soundfile as sf
from modeling.models import VanillaCNN, SimpleUNet

def si_sdr(ref, est, eps=1e-8):
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    alpha  = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    e_true = alpha * ref
    e_res  = est - e_true
    return 10 * np.log10((np.sum(e_true**2) + eps) / (np.sum(e_res**2) + eps))

def sdr(ref, est, eps=1e-8):
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    noise = ref - est
    return 10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum(noise ** 2) + eps))


def load_model(cfg, device):
    net_cfg = cfg['network']
    args = {
        'in_channels': net_cfg['in_channels'],
        'base_filters': net_cfg['base_filters'],
        'output_channels': net_cfg['output_channels'],
    }

    if net_cfg['model'] == "VanillaCNN":
        model = VanillaCNN(**args)
    elif net_cfg['model'] == "SimpleUNet":
        model = SimpleUNet(**args)
    else:
        raise ValueError(f"Unknown model: {net_cfg['model']}")

    ckpt_path = cfg.get('checkpoint_path', 'checkpoints/best_model.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval()

def main():
    cfg = yaml.safe_load(open('config.yaml'))
    device = torch.device(cfg.get('device', 'cpu'))

    N_FFT = 512
    HOP_LEN = 512
    WIN_LEN = 512
    window = torch.ones(WIN_LEN).to(device)

    model = load_model(cfg, device)

    spec_root = cfg['data']['root_dir']
    wav_root = os.path.dirname(spec_root)
    test_root = os.path.join(wav_root, 'test')

    all_scores = []

    for song_dir in sorted(glob.glob(f"{test_root}/*")):
        if not os.path.isdir(song_dir):
            continue

        mix_np, sr = sf.read(os.path.join(song_dir, 'mixture.wav'))
        if mix_np.ndim == 2:
            mix_np = mix_np.mean(axis=1)
        mix = torch.tensor(mix_np, dtype=torch.float32, device=device)

        spec = torch.stft(
            mix,
            n_fft=N_FFT,
            hop_length=HOP_LEN,
            win_length=WIN_LEN,
            window=window,
            return_complex=True,
            center = False
        )

        mag = spec.abs().unsqueeze(0).unsqueeze(0)
        phase = torch.angle(spec).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred_mag = model(mag)  # [1, 1, F+1, T] for vocals

        comp = pred_mag[0, 0] * torch.exp(1j * phase[0, 0])
        wav_est = torch.istft(
            comp,
            n_fft=N_FFT,
            hop_length=HOP_LEN,
            win_length=WIN_LEN,
            window=window,
            length=len(mix_np),
            center=False
        ).cpu().numpy()

        vocals_gt, _ = sf.read(os.path.join(song_dir, 'vocals.wav'))
        if vocals_gt.ndim == 2:
            vocals_gt = vocals_gt.mean(axis=1)

        if vocals_gt.shape[0] > wav_est.shape[0]:
            vocals_gt = vocals_gt[: wav_est.shape[0]]
        elif vocals_gt.shape[0] < wav_est.shape[0]:
            vocals_gt = np.pad(vocals_gt, (0, wav_est.shape[0] - vocals_gt.shape[0]))

        score_sisdr = si_sdr(vocals_gt, wav_est)
        score_sdr = sdr(vocals_gt, wav_est)
        all_scores.append(score_sdr)

        print(f"{os.path.basename(song_dir)} | SI-SDR: {score_sisdr:6.2f} dB | SDR: {score_sdr:6.2f} dB")

    print("\n=== Average SDR for Vocals ===")
    print(f"  vocals: {np.mean(all_scores):6.2f} dB")


if __name__ == "__main__":
    main()

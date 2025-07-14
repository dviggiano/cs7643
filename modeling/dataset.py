# modeling/dataset.py
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SpectrogramDataset(Dataset):
    """
    Loads .pt spectrograms for one subset (train/test) of MUSDB18HQ.
    Skips any clip directories missing required files.
    """

    STEM_ORDER    = ['bass', 'drums', 'other', 'vocals']
    REQUIRED_PTS  = ['mix.pt'] + [f'{s}.pt' for s in STEM_ORDER]

    def __init__(self, root_dir, subset='train', transform=None):
        self.base      = Path(root_dir) / subset
        self.transform = transform

        # gather only clips that have all required files
        self.clips = []
        for song_dir in self.base.iterdir():
            if not song_dir.is_dir(): 
                continue
            for clip_dir in song_dir.iterdir():
                if not clip_dir.is_dir(): 
                    continue

                missing = [f for f in self.REQUIRED_PTS
                           if not (clip_dir / f).exists()]
                if missing:
                    print(f"⚠️  Skipping {clip_dir} (missing {missing})")
                else:
                    self.clips.append(clip_dir)

        if not self.clips:
            raise RuntimeError(f"No valid clips found in {self.base}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_dir = self.clips[idx]

        # load mix (1 × F × T)
        mix = torch.load(clip_dir / 'mix.pt', weights_only = True)
        mix = mix.unsqueeze(0) if mix.ndim == 2 else mix

        # load stems and stack (4 × F × T)
        stems = []
        for s in self.STEM_ORDER:
            t = torch.load(clip_dir / f'{s}.pt', weights_only = True)
            t = t.unsqueeze(0) if t.ndim == 2 else t
            stems.append(t)
        target = torch.cat(stems, dim=0)

        # --- SIMPLE PADDING TO EVEN SIZES ---
        _, H, W = mix.shape
        pad_h = (0, 1) if H % 2 else (0, 0)
        pad_w = (0, 1) if W % 2 else (0, 0)
        # F.pad takes (left, right, top, bottom)
        mix    = F.pad(mix, (0, pad_w[1], 0, pad_h[1]))
        target = F.pad(target, (0, pad_w[1], 0, pad_h[1]))
        # --------------------------------------

        if self.transform:
            mix    = self.transform(mix)
            target = self.transform(target)

        return mix, target
#!/usr/bin/env python
# train.py

import os
import yaml
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modeling.dataset import SpectrogramDataset
from modeling.models import VanillaCNN, SimpleUNet
from tqdm import tqdm

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(config):
    # --- device setup ---
    device = torch.device(config.get('device', 'cpu'))

    # --- run directory & logging setup ---
    model_name = config['network']['model']
    base_dir   = config['logging']['base_dir']
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir    = os.path.join(base_dir, f"{model_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir)
    print(f"[INFO] TensorBoard events → {run_dir}")

    csv_path = os.path.join(run_dir, config['logging']['csv_filename'])
    print(f"[INFO] CSV log        → {csv_path}")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,val_loss\n')

    # --- dataset & dataloader setup ---
    ds_cfg = config['data']
    train_ds = SpectrogramDataset(ds_cfg['root_dir'], subset='train')
    val_ds   = SpectrogramDataset(ds_cfg['root_dir'], subset='test')

    batch_size = config['train']['batch_size']
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=ds_cfg['num_workers'],
        pin_memory=ds_cfg['pin_memory'],
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ds_cfg['num_workers'],
        pin_memory=ds_cfg['pin_memory'],
        persistent_workers=True
    )

    # --- model instantiation ---
    net_cfg = config['network']
    if net_cfg['model'] == "VanillaCNN":
        model = VanillaCNN(
            in_channels=net_cfg['in_channels'],
            base_filters=net_cfg['base_filters'],
            output_channels=net_cfg['output_channels']
        )
    elif net_cfg['model'] == "SimpleUNet":
        model = SimpleUNet(
            in_channels=net_cfg['in_channels'],
            base_filters=net_cfg['base_filters'],
            output_channels=net_cfg['output_channels']
        )
    else:
        raise ValueError(f"Unknown model: {net_cfg['model']}")

    model = model.to(device)

    # --- loss & optimizer & scheduler ---
    loss_type = config['loss']['type'].upper()
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'L1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    train_cfg = config['train']
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(train_cfg['learning_rate']),
        momentum=float(train_cfg['momentum']),
        weight_decay=float(train_cfg['weight_decay'])
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_cfg['milestones'],
        gamma=train_cfg['gamma']
    )

    # --- training loop ---
    best_val = float('inf')
    ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}", leave=False),
            1
        ):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        scheduler.step()
        avg_train = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)
        avg_val = val_loss / len(val_ds)

        # write epoch metrics
        writer.add_scalar('Loss/train_epoch', avg_train, epoch)
        writer.add_scalar('Loss/val_epoch',   avg_val,   epoch)

        with open(csv_path, 'a') as f:
            f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f}\n")

        print(f"→ Epoch {epoch}  train_loss: {avg_train:.4f}  val_loss: {avg_val:.4f}")

        # save best checkpoint
        if ds_cfg.get('save_best', False) and avg_val < best_val:
            best_val = avg_val
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val
            }
            torch.save(ckpt, os.path.join(ckpt_dir, 'best_model.pth'))
            print(f"✔️  Saved new best model (val_loss {best_val:.4f})")

    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)

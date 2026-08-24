
import dataloader_improved as dataloader
import time
import torch
from pathlib import Path
from unet import UNet
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.nn.functional as F
import numpy as np

def train_val_dataset(dataset, val_split=0.1):
    split_idx = int(len(dataset) * (1 - val_split))
    train_idx = list(range(0, split_idx))
    val_idx = list(range(split_idx, len(dataset)))
    datasets = {}
    datasets['train'] = torch.utils.data.Subset(dataset, train_idx)
    datasets['val'] = torch.utils.data.Subset(dataset, val_idx)
    return datasets

def compute_loss(model: UNet, device: torch.device, batch: dict, target_channels: int) -> tuple[float]:
    x, cond, batch_size = batch["target"].to(device), batch["input"].to(device), batch["target"].shape[0]
    t0 = torch.zeros(batch_size, dtype=torch.long, device=device)
    x0 = torch.zeros_like(x)
    prediction = model(x0, cond, t0)
    coarse = cond[:, :target_channels]
    coarse = F.interpolate(coarse, size=x.shape[-2:], mode='bilinear', align_corners=False)
    residual = x - coarse
    loss = F.mse_loss(prediction, residual)
    return loss

def one_step(
    model: UNet,
    device: torch.device,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    ema_model: AveragedModel,
    target_channels: int,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float]:

    with torch.autocast(device.type, enabled=(device.type == 'cuda')):
        loss = compute_loss(model, device, batch, target_channels)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    scaler.step(optimizer)
    scaler.update()

    ema_model.update_parameters(model)

    return loss

@torch.no_grad()
def validate(
    model: UNet,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    target_channels: int,
) -> float:
    #set model to evaluation mode
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        with torch.autocast(device.type, enabled=(device.type == 'cuda')):
            loss  = compute_loss(model, device, batch, target_channels)
        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)

def make_log_file(dir: Path = Path('logs'), filename: str = "training_log.txt") -> None:
    log_file = dir / filename
    if not log_file.exists():
        dir.mkdir(parents=True, exist_ok=True)
        log_file.touch()
    with open (log_file, 'w') as f:
        f.write("Epoch, Loss, Val Loss, LR\n")

    return log_file

def lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    eta_min_ratio: float = 0.0,
):

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main(
    data_dir: Path,
    checkpoint: str | None = None,
    batch_size: int = 16,
    val_split: float = 0.1,
    base_channels: int = 32,
    lr: float = 1e-4,
    min_lr: float = 1e-7,
    max_epochs: int = 5,
    checkpoint_output_dir: Path = Path("checkpoints"),
    ema_decay: float = 0.999, 
) -> None:

    _t_start = time.perf_counter()
    log_file = make_log_file()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)

    dataset = dataloader.ROMSDownscalingDataset(data_dir=data_dir)
    datasets = train_val_dataset(dataset, val_split=val_split)
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    train_time_indices = [dataset.valid_time_idx[i] for i in train_dataset.indices]
    dataset.input_stats = dataset._compute_stats(dataset.input_vars, coarsen=True, time_indices=train_time_indices)
    dataset.target_stats = dataset._compute_stats(dataset.target_vars, coarsen=False, time_indices=train_time_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    print(f'DATA LOADED ({time.perf_counter() - _t_start:.1f}s)')
    sample = dataset[0]
    cond_channels = sample["input"].shape[0]
    target_channels = sample["target"].shape[0]

    model = UNet(in_channels=target_channels, cond_channels=cond_channels, base_channels=base_channels).to(device)
    if device.type == 'cuda':
        model = torch.compile(model)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=ema_decay)).to(device)

    #Learning rate
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * max_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    start_epoch = 0
    best_val_loss = float("inf")

    for epoch in range(start_epoch, max_epochs):
        model.train() # set model to training mode
        epoch_train_loss = 0.0
        train_batches = 0

        # Training step 
        for batch in train_loader:
            loss = one_step(model, device, batch, optimizer, ema_model, target_channels, scaler)
            epoch_train_loss += loss.item()
            train_batches += 1
            scheduler.step()

        train_loss = epoch_train_loss / max(1, train_batches)

        #validate
        val_loss = validate(ema_model.module, device, val_loader, target_channels)

        #log training and validation lossa
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}, {train_loss:.10f}, {val_loss:.10f}, {optimizer.param_groups[0]['lr']:.10f}\n")
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {train_loss:.10f}, Learning Rate: {optimizer.param_groups[0]['lr']:.10f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_output_dir / "best_model.pt"
            checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
            # Unwrap torch.compile's OptimizedModule so state dict keys have no _orig_mod. prefix
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            raw_ema   = ema_model.module._orig_mod if hasattr(ema_model.module, '_orig_mod') else ema_model.module
            torch.save(
                {
                    'epoch': epoch + 1,
                    'best_val_loss': best_val_loss,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_model_state_dict': raw_ema.state_dict(),
                    'input_stats': dataset.input_stats,
                    'target_stats': dataset.target_stats,
                    'static_stats': dataset.static_stats,
                }, ckpt_path)

        ckpt_path = checkpoint_output_dir / f"model_epoch_last.pt"
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw_ema   = ema_model.module._orig_mod if hasattr(ema_model.module, '_orig_mod') else ema_model.module
        torch.save(
            {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_model_state_dict': raw_ema.state_dict(),
                'input_stats': dataset.input_stats,
                'target_stats': dataset.target_stats,
                'static_stats': dataset.static_stats,
            }, ckpt_path)
            


if __name__ == "__main__":  
    #main(Path('/home/mateuszm/downscaling_1/zarr/test.zarr'), val_split=0.1, checkpoint_output_dir=Path('/lustre/storeB/users/mateuszm/downscaling/exp1'), max_epochs=1000)
    main(Path('/home/mateuszm/downscaling_1/zarr/nk160_m71_20240501-20260531.zarr'), val_split=0.1, checkpoint_output_dir=Path('/lustre/storeB/users/mateuszm/downscaling/exp2'), max_epochs=1000)

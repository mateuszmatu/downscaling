
import dataloader as dataloader
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from unet import UNet
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.nn.functional as F
import numpy as np

def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = torch.utils.data.Subset(dataset, train_idx)
    datasets['val'] = torch.utils.data.Subset(dataset, val_idx)
    return datasets

def one_step(
    model: UNet,
    device: torch.device,
    batch: dict,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:

    x, cond, batch_size = batch["target"].to(device), batch["input"].to(device), batch["target"].shape[0]
    t0 = torch.zeros(batch_size, dtype=torch.long, device=device)
    x0 = torch.zeros_like(x)
    prediction = model(x0, cond, t0)
    loss = F.mse_loss(prediction, x)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss, prediction

def make_log_file(dir: Path = Path('logs'), filename: str = "training_log.txt") -> None:
    log_file = dir / filename
    if not log_file.exists():
        dir.mkdir(parents=True, exist_ok=True)
        log_file.touch()
    with open (log_file, 'w') as f:
        f.write("Epoch, Loss, Learning Rate\n")

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
    lr: float = 1e-3,
    min_lr: float = 1e-5,
    max_epochs: int = 5,
    checkpoint_output_dir: Path = Path("checkpoints"),
) -> None:

    log_file = make_log_file()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)

    dataset = dataloader.ROMSDownscalingDataset(data_dir=data_dir)
    datasets = train_val_dataset(dataset, val_split=val_split)
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available(), persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available(), persistent_workers=True)
    print('DATA LOADED')
    sample = dataset[0]
    cond_channels = sample["input"].shape[0]
    target_channels = sample["target"].shape[0]

    model = UNet(in_channels=target_channels, cond_channels=cond_channels, base_channels=base_channels).to(device)

    # Add EMA later

    #Learning rate
    total_epochs = 10
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * total_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    start_epoch = 0

    for epoch in range(start_epoch, max_epochs):
        # Training step 
        for batch in train_loader:
            loss, prediction = one_step(model, device, batch, optimizer)
            scheduler.step()

            with open(log_file, 'a') as f:
                f.write(f"{epoch+1}, {loss.item():.4f}, {optimizer.param_groups[0]['lr']:.6f}\n")
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

        ckpt_path = checkpoint_output_dir / f"model_epoch_{epoch+1}.pt"
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)
            


if __name__ == "__main__":  
    main(Path('/home/mateuszm/downscaling_1/zarr/test.zarr'), val_split=0.1, checkpoint_output_dir=Path('/lustre/storeB/users/mateuszm/downscaling/exp1'), max_epochs=1000)
    #main(Path('/home/mateuszm/downscaling_1/zarr/nk160_m71_20240501-20260531.zarr'), val_split=0.1, checkpoint_output_dir=Path('/lustre/storeB/users/mateuszm/downscaling/exp1'), max_epochs=1000)

import torch
from pathlib import Path
from unet import UNet
import xarray as xr
import numpy as np

def normalize(field: np.ndarray) -> np.ndarray:
    ## Maybe use mean of training dataset, not of field
    mean = np.mean(field)
    std = np.std(field)
    return (field - mean) / std

def denormalize(field: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (field * std) + mean

def resize_field(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(field).unsqueeze(0).unsqueeze(0).float()
    resized_tensor = torch.nn.functional.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
    return resized_tensor.squeeze().numpy()

def sample(cond: torch.Tensor, model: UNet) -> np.ndarray:
    batch_size = cond.shape[0]
    channels = 1
    out_h = cond.shape[2] * 5
    out_w = cond.shape[3] * 5
    x0 = torch.zeros((batch_size, channels, out_h, out_w), device=cond.device)
    t0 = torch.zeros(batch_size, dtype=torch.long, device=cond.device)
    x = model(x0, cond, t0)
    return x[0,0].detach().cpu().numpy()

def main(checkpoint_path: Path, input_netcdf: Path, output_netcdf: Path, base_channels: int = 32) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    for key, value in model_state.items():
        if key.endswith('inc.double_conv.0.weight'):
            inc_weight = value
        if key.endswith('outc.weight'):
            outc_weight = value
    input_channels = int(outc_weight.shape[0])
    total_in_channels = int(inc_weight.shape[1])
    cond_channels = total_in_channels - input_channels

    model = UNet(in_channels=input_channels, cond_channels=cond_channels, base_channels=base_channels).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Only first time for now
    ds = xr.open_dataset(input_netcdf).isel(depth=0, time=0)['temperature']
    coarse_ds = ds.coarsen(X=5, Y=5, boundary='trim').mean()
    coarse_field = np.nan_to_num(coarse_ds.values, nan=0.0, posinf=0.0, neginf=0.0)
    coarse_field = normalize(coarse_field)
    cond_tensor = torch.from_numpy(coarse_field).unsqueeze(0).unsqueeze(0).float().to(device)
    predicted_field = sample(cond_tensor, model)
    predicted_field = denormalize(predicted_field, np.mean(coarse_field), np.std(coarse_field))
    predicted_field = resize_field(predicted_field, ds.shape)

    # Save the predicted field to a new NetCDF file
    predicted_ds = xr.Dataset(
        {
            "predicted_temperature": (("Y", "X"), predicted_field)
        },
        coords={
            "Y": ds.coords["Y"],
            "X": ds.coords["X"]
        }
    )
    predicted_ds.to_netcdf(output_netcdf)



    print('passed')

if __name__ == "__main__":
    checkpoint_path = Path("/lustre/storeB/users/mateuszm/downscaling/exp1/model_epoch_1.pt")
    input_netcdf = Path('/home/mateuszm/downscaling_1/test_data/norkyst160_his_zdepth_20250101T00Z_m71_AN.nc')
    output_netcdf = Path('results/predicted_temperature.nc')
    main(checkpoint_path, input_netcdf, output_netcdf)
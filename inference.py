import torch
from pathlib import Path
from unet import UNet
import xarray as xr
import numpy as np

def normalize(field: np.ndarray, mean: float, std: float) -> np.ndarray:
    field = np.where(np.isfinite(field), field, mean)
    return (field - mean) / std

def denormalize(field: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (field * std) + mean

def resize_field(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(field).unsqueeze(0).unsqueeze(0).float()
    resized_tensor = torch.nn.functional.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
    return resized_tensor.squeeze().numpy()

def sample(cond: torch.Tensor, model: UNet, output_shape: tuple[int, int], out_channels: int) -> np.ndarray:
    batch_size = cond.shape[0]
    out_h, out_w = output_shape
    x0 = torch.zeros((batch_size, out_channels, out_h, out_w), device=cond.device)
    t0 = torch.zeros(batch_size, dtype=torch.long, device=cond.device)
    x = model(x0, cond, t0)
    return x[0,0].detach().cpu().numpy()

def main(checkpoint_path: Path, input_netcdf: Path, output_netcdf: Path, base_channels: int = 64) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['ema_model_state_dict']
    input_stats = checkpoint['input_stats']
    target_stats = checkpoint['target_stats']
    static_stats = checkpoint.get('static_stats', {})
    # Strip _orig_mod. prefix added by torch.compile if present
    model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
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

    input_var_name = next(iter(input_stats))
    target_var_name = next(iter(target_stats))
    static_var_name = next(iter(static_stats)) if static_stats else "h"

    input_mean = input_stats[input_var_name]["mean"]
    input_std = input_stats[input_var_name]["std"]
    target_mean = target_stats[target_var_name]["mean"]
    target_std = target_stats[target_var_name]["std"]
    static_mean = static_stats.get(static_var_name, {}).get("mean", 0.0)
    static_std = static_stats.get(static_var_name, {}).get("std", 1.0)

    full_ds = xr.open_dataset(input_netcdf).isel(depth=0)
    predicted_field = np.zeros((full_ds.time.size, full_ds.Y.size, full_ds.X.size), dtype=np.float32)
    predicted_residual = np.zeros((full_ds.time.size, full_ds.Y.size, full_ds.X.size), dtype=np.float32)
    first_ds = full_ds.isel(time=0)['temperature']
    first_coarse_ds = first_ds.coarsen(X=5, Y=5, boundary='trim').mean()
    coarse_800m_field = np.zeros((full_ds.time.size, first_coarse_ds.sizes['Y'], first_coarse_ds.sizes['X']), dtype=np.float32)
    for t in range(full_ds.time.size):
        ds = full_ds.isel(time=t)['temperature']
        coarse_ds = ds.coarsen(X=5, Y=5, boundary='trim').mean()
        coarse_field = coarse_ds.values
        coarse_800m_field[t] = coarse_ds.values
        coarse_field = normalize(coarse_field, input_mean, input_std)
        cond_tensor = torch.from_numpy(coarse_field).unsqueeze(0).unsqueeze(0).float().to(device)

        h_coarse = full_ds['h'].coarsen(X=5, Y=5, boundary='trim').mean().values
        h_coarse = normalize(h_coarse, static_mean, static_std)
        h_tensor = torch.from_numpy(h_coarse).unsqueeze(0).unsqueeze(0).float().to(device)

        cond_tensor = torch.cat((cond_tensor, h_tensor), dim=1)
        predicted_field_t = sample(cond_tensor, model, output_shape=ds.shape, out_channels=input_channels)
        coarse_target_norm = ((coarse_field * input_std) + input_mean - target_mean) / target_std
        coarse_resized = resize_field(coarse_target_norm, ds.shape)
        prediction_t = denormalize(predicted_field_t + coarse_resized, target_mean, target_std)
        residual_t = predicted_field_t * target_std
        prediction_t = np.where(np.isfinite(ds.values), prediction_t, np.nan)
        residual_t = np.where(np.isfinite(ds.values), residual_t, np.nan)
        predicted_field[t] = prediction_t
        predicted_residual[t] = residual_t

    # Save the predicted field to a new NetCDF file
    predicted_ds = xr.Dataset(
        {
            "predicted_temperature": (("time", "Y", "X"), predicted_field),
            "predicted_residual_temperature": (("time", "Y", "X"), predicted_residual),
            "input_temperature": (("time", "Y", "X"), full_ds['temperature'].values),
            "coarse_800m_temperature": (("time", "Y_800m", "X_800m"), coarse_800m_field),
        },
        coords={
            "time": full_ds.time.values,
            "Y": ds.coords["Y"].values,
            "X": ds.coords["X"].values,
            "Y_800m": first_coarse_ds.coords["Y"].values,
            "X_800m": first_coarse_ds.coords["X"].values,
        }
    )
    predicted_ds.to_netcdf(output_netcdf)



    print('passed')

def plot_predicted_field(ds, time_index: int) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(ds['input_temperature'].values[time_index], cmap='viridis', origin='lower')
    plt.title('Input Field')
    plt.colorbar()
    plt.subplot(1, 4, 2)
    plt.imshow(ds['coarse_800m_temperature'].values[time_index], cmap='viridis', origin='lower')
    plt.title('Coarse 800m Field')
    plt.colorbar()
    plt.subplot(1, 4, 3)
    plt.imshow(ds['predicted_temperature'].values[time_index], cmap='viridis', origin='lower')
    plt.title('Predicted Field')
    plt.colorbar()
    plt.tight_layout()
    plt.subplot(1, 4, 4)
    plt.imshow(ds['predicted_residual_temperature'].values[time_index], cmap='viridis', origin='lower')
    plt.title('Predicted Residual Field')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('results/predicted_field_comparison.png')

if __name__ == "__main__":
    checkpoint_path = Path("/lustre/storeB/users/mateuszm/downscaling/exp2/model_epoch_last.pt")
    input_netcdf = Path('/home/mateuszm/downscaling_1/test_data/norkyst160_his_zdepth_20250101T00Z_m71_AN.nc')
    output_netcdf = Path('results/predicted_temperature.nc')
    main(checkpoint_path, input_netcdf, output_netcdf)
    plot_predicted_field(xr.open_dataset('results/predicted_temperature.nc'), time_index=0)
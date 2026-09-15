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
    return x[0].detach().cpu().numpy()  # Returns (out_channels, out_h, out_w)

def main(checkpoint_path: Path, input_netcdf: list[Path], output_netcdf: Path, base_channels: int = 64) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['ema_model_state_dict']
    input_stats = checkpoint['input_stats']
    target_stats = checkpoint['target_stats']
    static_stats = checkpoint.get('static_stats', {})
    residuals = checkpoint.get('residuals', True)
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

    import re
    input_var_names = [re.sub('_0', '', var) for var in input_stats.keys()]
    target_var_names = [re.sub('_0', '', var) for var in target_stats.keys()]
    static_var_names = list(static_stats.keys()) if static_stats else []

    #still only surface
    if len(input_netcdf) == 1:
        ids = xr.open_dataset(input_netcdf[0]).isel(depth=0)
    elif len(input_netcdf) > 1:
        ids = xr.open_mfdataset(input_netcdf, combine='by_coords').isel(depth=0)
    else:
        raise ValueError("No input NetCDF files provided.")
    
    ref_shape = ids.isel(time=0)[target_var_names[0]].shape

    pfields = {var: np.zeros((ids.time.size, ids.Y.size, ids.X.size), dtype=np.float32) for var in target_var_names}
    presiduals = {var: np.zeros((ids.time.size, ids.Y.size, ids.X.size), dtype=np.float32) for var in target_var_names}
    pcoarse = {var: np.zeros((ids.time.size, ids.Y.size, ids.X.size), dtype=np.float32) for var in target_var_names}

    for t in range(ids.time.size):
        cond_parts = [] # what does this do?
        for var in input_var_names:
            tds = ids.isel(time=t)[var].coarsen(X=5, Y=5, boundary='trim').mean()
            imean = input_stats[var+'_0']["mean"] # temporary fix for the fact that input_stats keys have _0 suffix
            istd = input_stats[var+'_0']["std"]
            cfield = normalize(tds.values, imean, istd)
            cond_parts.append(torch.from_numpy(cfield).unsqueeze(0).float())

        for var in static_var_names:
            smean = static_stats[var]["mean"]
            sstd = static_stats[var]["std"]
            hds_da = ids[var]
            if 'time' in hds_da.dims:
                hds_da = hds_da.isel(time=t)
            hds = hds_da.coarsen(X=5, Y=5, boundary='trim').mean().values
            hfield = normalize(hds, smean, sstd)
            cond_parts.append(torch.from_numpy(hfield).unsqueeze(0).float())

        cond_tensor = torch.cat(cond_parts, dim=0).unsqueeze(0).to(device)

        shape = sample(cond_tensor, model, output_shape=ref_shape, out_channels=input_channels)

        for ch, var in enumerate(target_var_names):
            tds = ids.isel(time=t)[var]
            tds_coarse = tds.coarsen(X=5, Y=5, boundary='trim').mean()

            imean = input_stats[var+'_0']["mean"]
            istd = input_stats[var+'_0']["std"]
            tmean = target_stats[var+'_0']["mean"]
            tstd = target_stats[var+'_0']["std"]

            cfield = normalize(tds_coarse.values, imean, istd)
            cfield_norm = ((cfield * istd) + imean - tmean) / tstd
            cfield_resized_norm = resize_field(cfield_norm, tds.shape)
            cfield_resized = resize_field(tds.values, tds.shape)
            coarse_resized = resize_field(tds_coarse.values, tds.shape)

            model_ch = shape[ch]

            #check this if test
            if residuals:
                predicted_var = denormalize(model_ch + cfield_resized_norm, tmean, tstd)
                residual_var = model_ch * tstd
            else:
                predicted_var = denormalize(model_ch, tmean, tstd)
                residual_var = predicted_var - cfield_resized

            predicted_var = np.where(np.isfinite(tds.values), predicted_var, np.nan)
            residual_var = np.where(np.isfinite(tds.values), residual_var, np.nan)
            coarse_var = np.where(np.isfinite(tds.values), coarse_resized, np.nan)
            pfields[var][t] = predicted_var
            presiduals[var][t] = residual_var
            pcoarse[var][t] = coarse_var

        # Save the fields to NetCDF
        data_vars = {}
        for var in target_var_names:
            data_vars[f"predicted_{var}"] = (("time", "Y", "X"), pfields[var])
            data_vars[f"predicted_residual_{var}"] = (("time", "Y", "X"), presiduals[var])
            data_vars[f"coarse_{var}"] = (("time", "Y", "X"), pcoarse[var])

        for var in input_var_names:
            data_vars[f"input_{var}"] = (("time", "Y", "X"), ids[var].values)

        output_ds = xr.Dataset(
            data_vars,
            coords={
                "time": ids.time.values,
                "Y": ids.Y.values,
                "X": ids.X.values,
            })

        output_ds.to_netcdf(output_netcdf)


if __name__ == "__main__":
    import plot_func as pf
    checkpoint_path = Path("/lustre/storeB/users/mateuszm/downscaling/exp7/model_epoch_last.pt")
    #checkpoint_path = Path("/lustre/storeB/users/mateuszm/downscaling/exp7/best_model.pt")
    input_netcdf = [Path('/home/mateuszm/downscaling/test_data/norkyst160_his_zdepth_20260912T00Z_m71_AN.nc'),
                    Path('/home/mateuszm/downscaling/test_data/norkyst160_his_zdepth_20260913T00Z_m71_AN.nc'),
                    Path('/home/mateuszm/downscaling/test_data/norkyst160_his_zdepth_20260914T00Z_m71_AN.nc')]
    output_netcdf = Path('results/field.nc')
    #main(checkpoint_path, input_netcdf, output_netcdf)
    #ds_result = xr.open_dataset('results/predicted_temperature.nc')
    pf.plot_fields(output_netcdf, time_index=-1)
    pf.area_mean_timeseries(output_netcdf)
    pf.histogram(output_netcdf, vars=['abs_vel', 'u_eastward', 'v_northward'], bins=50, save_path='results/value_histogram.png')
    pf.scatter(output_netcdf, vars=['abs_vel', 'u_eastward', 'v_northward'], save_path='results/value_scatter.png')
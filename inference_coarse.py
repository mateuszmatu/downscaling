"""Downscale an already-coarse xarray dataset using a train.py checkpoint.

Example (surface velocities and bathymetry must already be on the same grid):

    from inference_coarse import downscale
    result = downscale(ds_800m, "checkpoints/best_model.pt")
    result.to_netcdf("prediction_160m.nc")

The default spatial dimensions are Y/X; y/Y and x/X are interchangeable when
resolving dimension names, and output retains the input's actual names.
Select depth/ensemble dimensions before calling, or pass spatial_dims for a
differently named grid. Variable names may
match the checkpoint (e.g. u_eastward_0) or omit its trailing "_0". Use
variable_map to override these aliases, e.g. {"u_eastward_0": "eastward"}.

Only load trusted checkpoints: train.py checkpoints contain Python objects.
Output masks are nearest-neighbor expansions of coarse finite-value masks,
not a true 160m coastline. Geographic coordinates are retained on the coarse
grid; fine-grid geolocation requires the corresponding grid/projection.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from unet import UNet


def _fine_coordinate(dataset: xr.Dataset, dim: str) -> np.ndarray:
    """Place five fine cell centers inside each regularly spaced coarse cell."""
    if dim not in dataset.coords:
        return (np.arange(dataset.sizes[dim] * 5) + 0.5) / 5 - 0.5
    coordinate = dataset.coords[dim]
    if coordinate.dims != (dim,) or not np.issubdtype(coordinate.dtype, np.number):
        raise ValueError(f"{dim!r} must have a numeric one-dimensional coordinate.")
    values = coordinate.values.astype(np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{dim!r} needs at least two finite coordinate values.")
    spacing = np.diff(values)
    if spacing[0] == 0 or not np.allclose(spacing, spacing[0], rtol=1e-5, atol=1e-8):
        raise ValueError(f"{dim!r} must be regularly spaced for 5x refinement.")
    return values[0] + ((np.arange(values.size * 5) + 0.5) / 5 - 0.5) * spacing[0]


def _statistics(checkpoint: dict, key: str) -> dict[str, dict[str, float]]:
    statistics = checkpoint[key]
    if not isinstance(statistics, dict):
        raise ValueError(f"Checkpoint {key!r} must be a variable-to-statistics dictionary.")
    for name, values in statistics.items():
        mean, std = float(values['mean']), float(values['std'])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization statistics for {name!r} in {key}.")
    return statistics


@torch.inference_mode()
def downscale(
    dataset: xr.Dataset,
    checkpoint_path: str | Path,
    *,
    variable_map: Mapping[str, str] | None = None,
    spatial_dims: tuple[str, str] = ('Y', 'X'),
    time_dim: str = 'time',
    device: str | torch.device | None = None,
    use_ema: bool = True,
) -> xr.Dataset:
    """Return 160m predictions from 800m inputs without further coarsening.

    Processes one timestep at a time, without DataLoader subprocesses. Output
    fields are predicted_<variable> and predicted_residual_<variable>
    (with trailing "_0" removed only from output names),
    in physical units on a grid with five times as many cells in each direction.
    coarse_<variable> fields preserve the original data on separate dimensions.
    A dataset without a time dimension returns a single spatial prediction.

    The caller must supply an 800m grid in the same physical conventions as
    training (eastward/northward surface velocities and bathymetry). No vector
    rotation or regridding between different projections is performed.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset.")
    spatial_dims = tuple(
        dim.swapcase()
        if dim in ('x', 'X', 'y', 'Y') and dim not in dataset.sizes
        and dim.swapcase() in dataset.sizes
        else dim
        for dim in spatial_dims
    )
    y_dim, x_dim = spatial_dims
    if y_dim == x_dim or time_dim in spatial_dims:
        raise ValueError("Spatial dimensions must be distinct from each other and time.")
    if any(dim not in dataset.sizes for dim in spatial_dims):
        raise ValueError(f"Dataset must contain spatial dimensions {spatial_dims}.")
    ny, nx = dataset.sizes[y_dim], dataset.sizes[x_dim]
    if min(ny * 5, nx * 5) < 16:
        raise ValueError("The fine grid must be at least 16x16 for the UNet pooling layers.")
    has_time = time_dim in dataset.sizes
    n_times = dataset.sizes[time_dim] if has_time else 1
    if n_times == 0:
        raise ValueError("Input dataset has no timesteps.")
    fine_y = _fine_coordinate(dataset, y_dim)
    fine_x = _fine_coordinate(dataset, x_dim)
    coarse_y, coarse_x = f'{y_dim}_800m', f'{x_dim}_800m'
    if any(dim in dataset.dims for dim in (coarse_y, coarse_x)):
        raise ValueError("Input dimensions conflict with output coarse-grid dimensions.")

    selected_device = torch.device(
        device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    input_stats = _statistics(checkpoint, 'input_stats')
    target_stats = _statistics(checkpoint, 'target_stats')
    static_stats = _statistics(checkpoint, 'static_stats')
    input_names, target_names = list(input_stats), list(target_stats)
    static_names = list(static_stats)
    state_key = 'ema_model_state_dict' if use_ema else 'model_state_dict'
    state = {
        key.removeprefix('_orig_mod.'): value
        for key, value in checkpoint[state_key].items()
    }
    initial_weight, output_weight = state['inc.double_conv.0.weight'], state['outc.weight']
    out_channels = output_weight.shape[0]
    cond_channels = initial_weight.shape[1] - out_channels
    if out_channels != len(target_names) or cond_channels != len(input_names) + len(static_names):
        raise ValueError("Checkpoint channel counts do not match its normalization statistics.")
    residuals = checkpoint['residuals']
    if not isinstance(residuals, bool):
        raise ValueError("Checkpoint residuals flag must be a boolean.")
    if residuals and input_names[:out_channels] != target_names:
        raise ValueError("Residual reconstruction requires matching leading input/target channels.")
    if not set(target_names).issubset(input_names):
        raise ValueError("Target variables must also be inputs to compute baselines and masks.")
    model = UNet(
        in_channels=out_channels, cond_channels=cond_channels,
        base_channels=initial_weight.shape[0],
    )
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model = model.to(selected_device).eval()

    names = list(dict.fromkeys(input_names + static_names))
    fields = {}
    variable_map = dict(variable_map or {})
    unknown = set(variable_map) - set(names)
    if unknown:
        raise ValueError(f"variable_map contains unknown checkpoint variables: {sorted(unknown)}")
    for name in names:
        alias = name.removesuffix('_0')
        variable = variable_map.get(name, name if name in dataset else alias)
        if variable not in dataset:
            raise ValueError(f"Missing {name!r}: provide {variable!r} or set variable_map.")
        field = dataset[variable]
        allowed_dims = set(spatial_dims) | {time_dim}
        if not set(spatial_dims).issubset(field.dims) or set(field.dims) - allowed_dims:
            raise ValueError(
                f"{variable!r} must have {spatial_dims} and optionally {time_dim!r}; "
                f"got {field.dims}. Select surface/depth/ensemble dimensions first."
            )
        if name in input_stats and has_time and time_dim not in field.dims:
            raise ValueError(f"Dynamic input {variable!r} needs dimension {time_dim!r}.")
        fields[name] = field

    def field_values(name: str, index: int) -> np.ndarray:
        field = fields[name]
        if time_dim in field.dims:
            field = field.isel({time_dim: index})
        return np.asarray(field.transpose(y_dim, x_dim).values, dtype=np.float32)

    def stats_tensor(statistics: dict, key: str) -> torch.Tensor:
        return torch.tensor(
            [statistics[name][key] for name in target_names],
            dtype=torch.float32, device=selected_device,
        )[None, :, None, None]

    target_mean, target_std = stats_tensor(target_stats, 'mean'), stats_tensor(target_stats, 'std')
    output_shape = (ny * 5, nx * 5)
    predictions = np.empty((n_times, out_channels, *output_shape), dtype=np.float32)
    corrections = np.empty_like(predictions)
    coarse_fields = np.empty((n_times, len(input_names), ny, nx), dtype=np.float32)
    for index in range(n_times):
        parts, masks = [], []
        for channel, name in enumerate(names):
            arr = field_values(name, index)
            statistics = input_stats[name] if name in input_stats else static_stats[name]
            finite = np.isfinite(arr)
            if name in input_stats:
                coarse_fields[index, channel] = np.where(finite, arr, np.nan)
                masks.append(torch.from_numpy(finite))
            normalized = (np.where(finite, arr, statistics['mean']) - statistics['mean']) / statistics['std']
            parts.append(torch.from_numpy(normalized.astype(np.float32)))
        cond = torch.stack(parts)[None].to(selected_device)
        x0 = torch.zeros((1, out_channels, *output_shape), device=selected_device)
        t0 = torch.zeros(1, dtype=torch.long, device=selected_device)
        prediction = model(x0, cond, t0)
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"Model returned non-finite predictions at timestep {index}.")
        predicted_physical = prediction * target_std + target_mean
        if residuals:
            coarse_normalized = F.interpolate(
                cond[:, :out_channels], size=output_shape, mode='bilinear', align_corners=False,
            )
            input_mean = stats_tensor(input_stats, 'mean')
            input_std = stats_tensor(input_stats, 'std')
            baseline = coarse_normalized * input_std + input_mean
            predicted_physical = prediction * target_std + baseline
        else:
            baseline_parts = []
            for name in target_names:
                if name not in input_names:
                    raise ValueError(f"Cannot compute residual output: target {name!r} is not an input.")
                channel = input_names.index(name)
                statistics = input_stats[name]
                baseline_parts.append(cond[:, channel] * statistics['std'] + statistics['mean'])
            baseline = F.interpolate(
                torch.stack(baseline_parts, dim=1), size=output_shape,
                mode='bilinear', align_corners=False,
            )
        target_masks = torch.stack([masks[input_names.index(name)] for name in target_names])
        valid = F.interpolate(
            target_masks[None].to(selected_device, dtype=torch.float32),
            size=output_shape, mode='nearest',
        ).bool()
        predictions[index] = predicted_physical.masked_fill(~valid, float('nan'))[0].cpu().numpy()
        corrections[index] = (predicted_physical - baseline).masked_fill(
            ~valid, float('nan'),
        )[0].cpu().numpy()
        print(f"Downscaled timestep {index + 1}/{n_times}: {ny}x{nx} -> {ny * 5}x{nx * 5}", flush=True)

    fine_dims = (time_dim, y_dim, x_dim) if has_time else spatial_dims
    coarse_dims = (time_dim, coarse_y, coarse_x) if has_time else (coarse_y, coarse_x)
    output = xr.Dataset(attrs={
        'checkpoint': str(checkpoint_path), 'checkpoint_weights': 'ema' if use_ema else 'model',
        'input_resolution_m': 800, 'output_resolution_m': 160,
        'residual_reconstruction': int(residuals),
        'mask_description': 'Nearest-neighbor coarse finite-value masks; no fine coastline supplied.',
    })
    output = output.assign_coords({
        y_dim: fine_y, x_dim: fine_x,
        coarse_y: dataset[y_dim].values if y_dim in dataset.coords else np.arange(ny),
        coarse_x: dataset[x_dim].values if x_dim in dataset.coords else np.arange(nx),
    })
    if has_time:
        output = output.assign_coords({
            time_dim: dataset[time_dim] if time_dim in dataset.coords else np.arange(n_times),
        })
    for dim, coarse_dim in zip(spatial_dims, (coarse_y, coarse_x)):
        if dim in dataset.coords:
            output[dim].attrs = dataset[dim].attrs.copy()
            output[coarse_dim].attrs = dataset[dim].attrs.copy()
    for channel, name in enumerate(target_names):
        attrs = {key: value for key, value in fields[name].attrs.items()
                 if key not in ('coordinates', 'grid_mapping')}
        for prefix, values in [('predicted_', predictions), ('predicted_residual_', corrections)]:
            output[prefix + name.removesuffix('_0')] = xr.DataArray(
                values[:, channel] if has_time else values[0, channel],
                dims=fine_dims, attrs=attrs,
            )
    for channel, name in enumerate(input_names):
        output['coarse_' + name.removesuffix('_0')] = xr.DataArray(
            coarse_fields[:, channel] if has_time else coarse_fields[0, channel],
            dims=coarse_dims, attrs=fields[name].attrs.copy(),
        )
    for name in ('lon', 'lat', 'longitude', 'latitude'):
        if name in dataset and set(dataset[name].dims).issubset(set(spatial_dims)):
            rename_dims = {dim: coarse_dim for dim, coarse_dim in
                           zip(spatial_dims, (coarse_y, coarse_x)) if dim in dataset[name].dims}
            output['coarse_' + name] = dataset[name].rename(rename_dims)
    return output

if __name__ == "__main__":
    import plot_func
    file = 'https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_his_files/2026/10/03/norkyst800_his_zdepth_20261003T00Z_m00_AN.nc'
    file = 'https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_his_files/2026/06/06/norkyst800_his_zdepth_20260606T00Z_m71_AN.nc'
    ds_800m = xr.open_dataset(file).isel(time=slice(0,3),depth=0, X=slice(1660,1800), Y=slice(660, 800))

    checkpoint_path = Path("/lustre/storeB/users/mateuszm/downscaling/exp11/best_model.pt")
    result = downscale(
    ds_800m,
    checkpoint_path,
    spatial_dims=("X", "Y"),
    variable_map={
        "u_eastward_0": "u_eastward",
        "v_northward_0": "v_northward",
        "h_0": "h",
    },
)
    result.to_netcdf('results/downscaled_field.nc')
    plot_func.plot_fields('results/downscaled_field.nc', time_index=0, truth=False)

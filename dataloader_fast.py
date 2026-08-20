from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


def _block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average last two spatial dims by `factor`.
    Matches xarray coarsen(..., boundary='trim').mean().
    """
    ny, nx = arr.shape[-2], arr.shape[-1]
    ny_c = ny // factor
    nx_c = nx // factor
    a = arr[..., :ny_c * factor, :nx_c * factor]
    shape = arr.shape[:-2] + (ny_c, factor, nx_c, factor)
    return a.reshape(shape).mean(axis=(-3, -1))


class ROMSDownscalingDataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
            input_vars: list[str] = ['temperature_0'],
            target_vars: list[str] = ['temperature_0'],
            static_vars: list[str] = ['h'],
            coarsen_factor: int = 5,
    ) -> None:

        self.data_dir = data_dir
        self.coarsen_factor = coarsen_factor
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.static_vars = static_vars
        self.y_dim = 'Y'
        self.x_dim = 'X'

        # Open with native chunks to avoid chunk-mismatch warnings
        ds_zarr = xr.open_zarr(data_dir, consolidated=False)
        if 'ensemble' in ds_zarr.dims:
            ds_zarr = ds_zarr.isel(ensemble=0)

        variable_names = list(ds_zarr.attrs.get('variables', []))
        self.var_to_idx = {var: i for i, var in enumerate(variable_names)}
        ny, nx = (int(v) for v in ds_zarr.attrs.get('field_shape'))
        self.field_shape = (ny, nx)

        # Unique variables to load (dynamic + static), preserving order
        all_vars = list(dict.fromkeys(input_vars + target_vars + (static_vars or [])))
        var_indices = [self.var_to_idx[v] for v in all_vars]
        self._all_vars = all_vars
        self._var_to_arr_idx = {v: i for i, v in enumerate(all_vars)}

        # Load all needed data in one zarr read: (time, n_vars, cell) -> (T, V, Y, X)
        print(f"Loading {len(all_vars)} variable(s) from zarr...", flush=True)
        data_da = ds_zarr['data'].isel(variable=var_indices).transpose('time', 'variable', 'cell')
        raw = data_da.load().values.astype(np.float32)  # (T, V, C)
        self._data = raw.reshape(raw.shape[0], len(all_vars), ny, nx)  # (T, V, Y, X)
        self.total_times = self._data.shape[0]
        print("Done loading.", flush=True)

        self.valid_time_idx = self._valid_time_idx()
        self.input_stats = self._compute_stats(self.input_vars, coarsen=True)
        self.target_stats = self._compute_stats(self.target_vars, coarsen=False)
        self.static_stats = {}
        self.static_tensor = self._static_vars() if self.static_vars else None

    def __len__(self) -> int:
        return len(self.valid_time_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self.valid_time_idx):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self.valid_time_idx)}")

        t = self.valid_time_idx[idx]

        target_parts = []
        for var in self.target_vars:
            arr = self._data[t, self._var_to_arr_idx[var]]  # (Y, X)
            arr = self._normalize(arr, self.target_stats[var]['mean'], self.target_stats[var]['std'])
            target_parts.append(torch.from_numpy(arr[np.newaxis]))

        input_parts = []
        for var in self.input_vars:
            arr = self._data[t, self._var_to_arr_idx[var]]  # (Y, X)
            arr_c = _block_mean(arr, self.coarsen_factor)   # (Y_c, X_c)
            arr_c = self._normalize(arr_c, self.input_stats[var]['mean'], self.input_stats[var]['std'])
            input_parts.append(torch.from_numpy(arr_c[np.newaxis]))

        target_tensor = torch.cat(target_parts, dim=0)
        input_tensor = torch.cat(input_parts, dim=0)
        if self.static_tensor is not None:
            input_tensor = torch.cat([input_tensor, self.static_tensor], dim=0)

        return {'input': input_tensor, 'target': target_tensor, 'time_idx': idx}

    # -------------------------------------------------------------------------

    def _normalize(self, arr: np.ndarray, mean: float, std: float) -> np.ndarray:
        arr = np.where(np.isfinite(arr), arr, mean)
        return ((arr - mean) / std).astype(np.float32)

    def _valid_time_idx(self) -> list[int]:
        """Vectorised: find timesteps where at least one target cell is finite."""
        target_indices = [self._var_to_arr_idx[v] for v in self.target_vars]
        target_data = self._data[:, target_indices, :, :]   # (T, n_target, Y, X)
        has_finite = np.isfinite(target_data).any(axis=(1, 2, 3))  # (T,)
        return list(np.where(has_finite)[0])

    def _compute_stats(self, var_names: list[str], coarsen: bool = False,
                       time_indices: list[int] | None = None) -> dict[str, dict[str, float]]:
        """Compute mean/std over the given time indices (positions in self._data)."""
        indices = self.valid_time_idx if time_indices is None else list(time_indices)
        stats = {}
        for var in var_names:
            arr = self._data[indices, self._var_to_arr_idx[var]]  # (T, Y, X)
            if coarsen:
                arr = _block_mean(arr, self.coarsen_factor)       # (T, Y_c, X_c)
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                mean, std = 0.0, 1.0
            else:
                mean = float(valid.mean())
                std = max(float(valid.std()), 1e-8)
            stats[var] = {'mean': mean, 'std': std}
        return stats

    def _static_vars(self) -> torch.Tensor:
        tensors = []
        for var in self.static_vars:
            arr = self._data[0, self._var_to_arr_idx[var]]  # (Y, X) — time=0 for static
            arr_c = _block_mean(arr, self.coarsen_factor)    # (Y_c, X_c)
            valid = arr_c[np.isfinite(arr_c)]
            if valid.size == 0:
                mean, std = 0.0, 1.0
            else:
                mean = float(valid.mean())
                std = max(float(valid.std()), 1e-8)
            self.static_stats[var] = {'mean': mean, 'std': std}
            arr_c = self._normalize(arr_c, mean, std)
            tensors.append(torch.from_numpy(arr_c[np.newaxis]))  # (1, Y_c, X_c)
        return torch.cat(tensors, dim=0)  # (n_static, Y_c, X_c)


if __name__ == "__main__":
    dataset = ROMSDownscalingDataset(
        Path('/home/mateuszm/downscaling_1/zarr/test.zarr'),
    )
    print(dataset.__len__())
    print(dataset._static_vars())
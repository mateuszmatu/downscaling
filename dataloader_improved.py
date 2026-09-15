from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


def _block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average the last two spatial dims by `factor`.

    Equivalent to xarray's coarsen(..., boundary='trim').mean():
    trims the array so it is divisible by factor, then averages
    each (factor x factor) block.  NaN propagates (same as np.mean).
    """
    ny, nx = arr.shape[-2], arr.shape[-1]
    ny_c = ny // factor
    nx_c = nx // factor
    a = arr[..., :ny_c * factor, :nx_c * factor]
    shape = arr.shape[:-2] + (ny_c, factor, nx_c, factor)
    return a.reshape(shape).mean(axis=(-3, -1))


class ROMSDownscalingDataset(Dataset):
    """Drop-in replacement for the original ROMSDownscalingDataset.

    Key difference: all needed variables are loaded from the zarr store
    into a single numpy array during __init__.  Every subsequent operation
    (valid_time_idx, compute_stats, __getitem__) works entirely in RAM,
    eliminating per-sample disk IO that was the dominant bottleneck.
    """

    def __init__(
            self,
            data_dir: Path,
            input_vars: list[str] = ['u_eastward_0', 'v_northward_0'], #['temperature_0', 'u_eastward_0', 'v_northward_0', 'salinity_0'],
            target_vars: list[str] = ['u_eastward_0', 'v_northward_0'], #['temperature_0', 'u_eastward_0', 'v_northward_0', 'salinity_0'],
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

        # Open with native chunks to avoid the chunk-mismatch warning/penalty
        ds_zarr = xr.open_zarr(data_dir, consolidated=False)
        if 'ensemble' in ds_zarr.dims:
            ds_zarr = ds_zarr.isel(ensemble=0)

        self.variable_names = list(ds_zarr.attrs.get('variables', []))
        self.var_to_idx = {var: i for i, var in enumerate(self.variable_names)}
        ny, nx = (int(v) for v in ds_zarr.attrs.get('field_shape'))
        self.field_shape = (ny, nx)

        # Collect every unique variable we need (dynamic + static), preserving order
        all_vars = list(dict.fromkeys(input_vars + target_vars + (static_vars or [])))
        var_indices = [self.var_to_idx[v] for v in all_vars]
        self._all_vars = all_vars
        self._var_to_arr_idx = {v: i for i, v in enumerate(all_vars)}

        # ── Single zarr read: (time, n_vars, cell) → (T, V, Y, X) ──────────
        print(f"Loading {len(all_vars)} variable(s) from zarr...", flush=True)
        data_da = ds_zarr['data'].isel(variable=var_indices).transpose('time', 'variable', 'cell')
        raw = data_da.load().values.astype(np.float32)   # (T, V, C)
        self._data = raw.reshape(raw.shape[0], len(all_vars), ny, nx)  # (T, V, Y, X)
        self.total_times = self._data.shape[0]
        print(f"Loaded {self.total_times} timesteps into RAM "
              f"({self._data.nbytes / 1024**2:.0f} MB).", flush=True)

        self.valid_time_idx = self._valid_time_idx()
        self.input_stats = self._compute_stats(self.input_vars, coarsen=True)
        self.target_stats = self._compute_stats(self.target_vars, coarsen=False)
        self.static_stats = {}
        self.static_tensor = self._static_vars() if self.static_vars else None

    # ── Public interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.valid_time_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self.valid_time_idx):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of length {len(self.valid_time_idx)}"
            )

        t = self.valid_time_idx[idx]

        target_parts = []
        target_mask_parts = []
        for var in self.target_vars:
            arr = self._data[t, self._var_to_arr_idx[var]]          # (Y, X)
            mask = np.isfinite(arr).astype(np.float32)
            arr = self._normalize(arr, self.target_stats[var]['mean'],
                                  self.target_stats[var]['std'])
            target_parts.append(torch.from_numpy(arr[np.newaxis]))  # (1, Y, X)
            target_mask_parts.append(torch.from_numpy(mask[np.newaxis]))

        input_parts = []
        for var in self.input_vars:
            arr = self._data[t, self._var_to_arr_idx[var]]          # (Y, X)
            arr_c = _block_mean(arr, self.coarsen_factor)            # (Y_c, X_c)
            arr_c = self._normalize(arr_c, self.input_stats[var]['mean'],
                                    self.input_stats[var]['std'])
            input_parts.append(torch.from_numpy(arr_c[np.newaxis])) # (1, Y_c, X_c)

        target_tensor = torch.cat(target_parts, dim=0)
        target_mask_tensor = torch.cat(target_mask_parts, dim=0)
        input_tensor = torch.cat(input_parts, dim=0)
        if self.static_tensor is not None:
            input_tensor = torch.cat([input_tensor, self.static_tensor], dim=0)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'target_mask': target_mask_tensor,
            'time_idx': idx,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize(self, arr: np.ndarray, mean: float, std: float) -> np.ndarray:
        arr = np.where(np.isfinite(arr), arr, mean)
        return ((arr - mean) / std).astype(np.float32)

    def _valid_time_idx(self) -> list[int]:
        """Vectorised: find timesteps where at least one target cell is finite."""
        target_indices = [self._var_to_arr_idx[v] for v in self.target_vars]
        target_data = self._data[:, target_indices, :, :]       # (T, n_target, Y, X)
        has_finite = np.isfinite(target_data).any(axis=(1, 2, 3))  # (T,)
        return list(np.where(has_finite)[0])

    def _compute_stats(self, var_names: list[str], coarsen: bool = False,
                       time_indices: list[int] | None = None) -> dict[str, dict[str, float]]:
        """Compute mean/std using the same moment-based formula as the Zarr loader.

        This matches the statistics used in dataloader_zarr.py precisely:
            mean = sum(x) / N
            variance = sum(x^2) / N - mean^2
            std = sqrt(max(variance, 0))
        """
        indices = self.valid_time_idx if time_indices is None else list(time_indices)
        stats: dict[str, dict[str, float]] = {}

        for var in var_names:
            total_count = 0
            total_sum = 0.0
            total_sum_sq = 0.0

            for idx in indices:
                arr = self._data[idx, self._var_to_arr_idx[var]]  # (Y, X)
                if coarsen:
                    arr = _block_mean(arr, self.coarsen_factor)   # (Y_c, X_c)

                valid_values = arr[np.isfinite(arr)].astype(np.float64)
                if valid_values.size == 0:
                    continue

                total_count += int(valid_values.size)
                total_sum += float(valid_values.sum())
                total_sum_sq += float(np.square(valid_values).sum())

            if total_count == 0:
                mean = 0.0
                std = 1.0
            else:
                mean = total_sum / total_count
                variance = max((total_sum_sq / total_count) - (mean * mean), 0.0)
                std = float(np.sqrt(variance))

            stats[var] = {'mean': float(mean), 'std': max(std, 1e-8)}
        return stats

    def _static_vars(self) -> torch.Tensor:
        tensors = []
        for var in self.static_vars:
            arr = self._data[0, self._var_to_arr_idx[var]]  # (Y, X) — static, use time=0
            arr_c = _block_mean(arr, self.coarsen_factor)    # (Y_c, X_c)

            valid_values = arr_c[np.isfinite(arr_c)].astype(np.float64)
            if valid_values.size == 0:
                mean = 0.0
                std = 1.0
            else:
                mean = float(valid_values.mean())
                variance = float(np.square(valid_values).mean() - (mean * mean))
                std = max(float(np.sqrt(max(variance, 0.0))), 1e-8)

            self.static_stats[var] = {'mean': mean, 'std': std}
            arr_c = self._normalize(arr_c, mean, std)
            tensors.append(torch.from_numpy(arr_c[np.newaxis]))  # (1, Y_c, X_c)
        return torch.cat(tensors, dim=0)                         # (n_static, Y_c, X_c)

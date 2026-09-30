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
            data_dir: Path | list[Path],
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

        paths = [data_dir] if isinstance(data_dir, (str, Path)) else list(data_dir)

        # Collect every unique variable we need (dynamic + static), preserving order
        all_vars = list(dict.fromkeys(input_vars + target_vars + (static_vars or [])))
        self._all_vars = all_vars
        self._var_to_arr_idx = {v: i for i, v in enumerate(all_vars)}

        # Open with native chunks to avoid the chunk-mismatch warning/penalty
        stores = []
        for p in paths:
            ds_zarr = xr.open_zarr(p, consolidated=False)
            if 'ensemble' in ds_zarr.dims:
                ds_zarr = ds_zarr.isel(ensemble=0)
            stores.append(ds_zarr)

        shapes = [(int(z.sizes['time']), *(int(v) for v in z.attrs.get('field_shape'))) for z in stores]
        if len(set(shapes)) != 1:
            raise ValueError(f"All sources must have the same (time, Y, X) shape, got {dict(zip(paths, shapes))}")
        n_times, ny, nx = shapes[0]
        self.field_shape = (ny, nx)
        self.total_times = n_times

        # ── One zarr read per source into a preallocated (S, T, V, Y, X) array ──
        self._data = np.empty((len(stores), n_times, len(all_vars), ny, nx), dtype=np.float32)
        for s, (p, ds_zarr) in enumerate(zip(paths, stores)):
            print(f"Loading {len(all_vars)} variable(s) from {p}...", flush=True)
            var_to_idx = {var: i for i, var in enumerate(ds_zarr.attrs.get('variables', []))}
            data_da = ds_zarr['data'].isel(variable=[var_to_idx[v] for v in all_vars]).transpose('time', 'variable', 'cell')
            self._data[s] = data_da.load().values.reshape(n_times, len(all_vars), ny, nx)
        self.n_sources = len(stores)
        print(f"Loaded {self.n_sources} source(s) x {self.total_times} timesteps into RAM "
              f"({self._data.nbytes / 1024**2:.0f} MB).", flush=True)

        # Each sample is a (source, time) pair
        self.samples = self._valid_samples()
        self.sample_source = [s for s, _ in self.samples]
        self.valid_time_idx = [t for _, t in self.samples]
        self.input_stats = self._compute_stats(self.input_vars, coarsen=True)
        self.target_stats = self._compute_stats(self.target_vars, coarsen=False)
        self.static_stats = {}
        self.static_tensor = self._static_vars() if self.static_vars else None

    # ── Public interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of length {len(self.samples)}"
            )

        s, t = self.samples[idx]

        target_parts = []
        target_mask_parts = []
        for var in self.target_vars:
            arr = self._data[s, t, self._var_to_arr_idx[var]]       # (Y, X)
            mask = np.isfinite(arr).astype(np.float32)
            arr = self._normalize(arr, self.target_stats[var]['mean'],
                                  self.target_stats[var]['std'])
            target_parts.append(torch.from_numpy(arr[np.newaxis]))  # (1, Y, X)
            target_mask_parts.append(torch.from_numpy(mask[np.newaxis]))

        input_parts = []
        for var in self.input_vars:
            arr = self._data[s, t, self._var_to_arr_idx[var]]       # (Y, X)
            arr_c = _block_mean(arr, self.coarsen_factor)            # (Y_c, X_c)
            arr_c = self._normalize(arr_c, self.input_stats[var]['mean'],
                                    self.input_stats[var]['std'])
            input_parts.append(torch.from_numpy(arr_c[np.newaxis])) # (1, Y_c, X_c)

        target_tensor = torch.cat(target_parts, dim=0)
        target_mask_tensor = torch.cat(target_mask_parts, dim=0)
        input_tensor = torch.cat(input_parts, dim=0)
        if self.static_tensor is not None:
            input_tensor = torch.cat([input_tensor, self.static_tensor[s]], dim=0)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'target_mask': target_mask_tensor,
            'time_idx': idx,
            'source_idx': s,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize(self, arr: np.ndarray, mean: float, std: float) -> np.ndarray:
        arr = np.where(np.isfinite(arr), arr, mean)
        return ((arr - mean) / std).astype(np.float32)

    def _valid_samples(self) -> list[tuple[int, int]]:
        """Vectorised: find (source, time) pairs where at least one target cell is finite."""
        target_indices = [self._var_to_arr_idx[v] for v in self.target_vars]
        target_data = self._data[:, :, target_indices]               # (S, T, n_target, Y, X)
        has_finite = np.isfinite(target_data).any(axis=(2, 3, 4))    # (S, T)
        return [(int(s), int(t)) for s, t in zip(*np.where(has_finite))]

    def _compute_stats(self, var_names: list[str], coarsen: bool = False,
                       sample_indices: list[int] | None = None) -> dict[str, dict[str, float]]:
        """Compute mean/std using the same moment-based formula as the Zarr loader.

        This matches the statistics used in dataloader_zarr.py precisely:
            mean = sum(x) / N
            variance = sum(x^2) / N - mean^2
            std = sqrt(max(variance, 0))
        """
        indices = range(len(self.samples)) if sample_indices is None else sample_indices
        stats: dict[str, dict[str, float]] = {}

        for var in var_names:
            total_count = 0
            total_sum = 0.0
            total_sum_sq = 0.0

            for idx in indices:
                s, t = self.samples[idx]
                arr = self._data[s, t, self._var_to_arr_idx[var]]  # (Y, X)
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
        """Static fields per source, normalized with stats pooled over all sources."""
        tensors = []
        for var in self.static_vars:
            arr = self._data[:, 0, self._var_to_arr_idx[var]]  # (S, Y, X) — static, use time=0
            arr_c = _block_mean(arr, self.coarsen_factor)       # (S, Y_c, X_c)

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
            tensors.append(torch.from_numpy(arr_c[:, np.newaxis]))  # (S, 1, Y_c, X_c)
        return torch.cat(tensors, dim=1)                            # (S, n_static, Y_c, X_c)

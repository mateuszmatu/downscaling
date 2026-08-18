from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

def open_zarr_dataset(
        path: Path
        ) -> xr.Dataset:
    """Open a zarr dataset and return it as an xarray Dataset."""
    ds = xr.open_zarr(path, consolidated=False, chunks={"time": 1, "Y": 100, "X": 100})
    return ds

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

        self.dataset = open_zarr_dataset(data_dir)
        if 'ensemble' in self.dataset.dims:
            self.dataset = self.dataset.isel(ensemble=0)

        self.variable_names = list(self.dataset.attrs.get('variables', []))
        self.var_to_idx = {var: i for i, var in enumerate(self.variable_names)}

        self.field_shape = (int(self.dataset.attrs.get('field_shape')[0]), int(self.dataset.attrs.get('field_shape')[1]))
        self.y_dim = 'Y'
        self.x_dim = 'X'

        self.total_times = int(self.dataset.sizes['time'])
        self.valid_time_idx=self._valid_time_idx()
        self.input_stats = self._compute_stats(self.input_vars, coarsen=True)
        self.target_stats = self._compute_stats(self.target_vars, coarsen=False)
        self.static_stats = {}
        self.static_tensor = self._static_vars() if self.static_vars else None
        
    def __len__(self) -> int:
        return len(self.valid_time_idx)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx = self.total_times + idx
        if idx < 0 or idx >= self.total_times:
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {self.total_times}")

        ds_idx = self.valid_time_idx[idx]
        target_ds = self._build_dataset(ds_idx, self.target_vars)
        input_high_res_ds = self._build_dataset(ds_idx, self.input_vars)
        input_low_res_ds = self._coarsen_dataset(input_high_res_ds, self.coarsen_factor)

        input_tensor = self._create_tensor(input_low_res_ds, self.input_stats)
        input_tensor = torch.cat((input_tensor, self.static_tensor), dim=0)
        target_tensor = self._create_tensor(target_ds, self.target_stats)

        return {'input': input_tensor, 'target': target_tensor, 'time_idx': idx}

    def _create_tensor(self, ds: xr.Dataset, stats: dict[str, dict[str, float]]) -> torch.Tensor:
        # Normalized the tensor for each variable using the provided mean and std.
        tensors = []
        for var in ds.data_vars:
            da = ds[var]
            if 'time' in da.dims:
                da = da.isel(time=0) if da.sizes['time'] == 1 else da.squeeze('time', drop=True)
            if 'depth' in da.dims:
                da = da.isel(depth=0)
            elif 'depth' not in da.dims:
                da = da.expand_dims('depth', axis=0)
            da = da.transpose('depth', self.y_dim, self.x_dim)
            array = np.asarray(da.values, dtype=np.float32)
            tensor = torch.from_numpy(array)
            mean, std = stats[var]['mean'], stats[var]['std']
            tensor = torch.nan_to_num(tensor, nan=mean, posinf=mean, neginf=mean)
            tensor = (tensor - mean) / std

            tensors.append(tensor)

        return torch.cat(tensors, dim=0)

    def _compute_stats(self, var_names: list[str], coarsen: bool = False, time_indices: list[int] | None = None) -> dict[str, dict[str, float]]:
        # Compute mean and std for each variable, used to normalize later. 
        stats = {}
        indices = self.valid_time_idx if time_indices is None else time_indices
        for var in var_names:
            count = 0
            sum = 0.0
            sum_sq = 0.0

            for t in indices:
                da = self._build_dataset(t, [var])
                if coarsen:
                    da = self._coarsen_dataset(da, self.coarsen_factor)

                values = np.asarray(da[var].values, dtype=np.float64)
                mask = np.isfinite(values)

                valid_values = values[mask]
                count += int(valid_values.size)
                sum += float(valid_values.sum())
                sum_sq += float((valid_values ** 2).sum())

            if count == 0:
                mean = 0.0
                std = 1.0
            else:
                mean = sum / count
                variance = max((sum_sq / count) - (mean ** 2), 0.0)
                std = max(np.sqrt(variance), 1e-8)

            stats[var] = {'mean': mean, 'std': std}
        return stats

    def _coarsen_dataset(self, ds: xr.Dataset, factor: int = 5) -> xr.Dataset:
        return ds.coarsen({self.y_dim: factor, self.x_dim: factor}, boundary='trim').mean()

    def _build_dataset(self, time_idx: int, var_names: list[str]) -> xr.Dataset:
        data_vars = {}
        for var in var_names:
            ds = self.dataset.isel(time=time_idx)
            variable_index = self.var_to_idx[var]
            da = ds['data'].isel({'variable': variable_index})
            values = np.asarray(da.load().values, dtype=np.float32)
            ny, nx = self.field_shape
            grid = values.reshape(ny, nx)
            data_vars[var] = xr.DataArray(grid, dims=(self.y_dim, self.x_dim), name=var)
        return xr.Dataset(data_vars=data_vars)  

    def _valid_time_idx(self) -> list[int]:
        valid_times = []
        for t in range(self.total_times):
            has_valid_target = False
            for var in self.target_vars:
                ds = self.dataset.isel(time=t)
                variable_index = self.var_to_idx[var]
                ds = ds['data'].isel(variable=variable_index)
                values = np.asarray(ds.values)
                if np.isfinite(values).any():
                    has_valid_target = True
                    break
            if has_valid_target:
                valid_times.append(t)

        return valid_times

    def _static_vars(self) -> torch.Tensor:
        tensors = []
        static_ds = self._build_dataset(0, self.static_vars)
        for var in self.static_vars:
            ds = static_ds[var]
            ds_coarse = self._coarsen_dataset(ds.to_dataset(name=var), self.coarsen_factor)
            values = np.asarray(ds_coarse[var].values, dtype=np.float64)
            mask = np.isfinite(values)
            valid_values = values[mask]
            count = int(valid_values.size)
            sum = float(valid_values.sum())
            sum_sq = float((valid_values ** 2).sum())
            if count == 0:
                mean = 0.0
                std = 1.0
            else:
                mean = sum / count
                variance = max((sum_sq / count) - (mean ** 2), 0.0)
                std = np.sqrt(variance)
            self.static_stats[var] = {'mean': mean, 'std': std}
            static_stats = {var: {'mean': mean, 'std': std}}
            tensor = self._create_tensor(ds_coarse, static_stats)
            tensors.append(tensor)

        return torch.cat(tensors, dim=0)


if __name__ == "__main__":
    dataset = ROMSDownscalingDataset(
        Path('/home/mateuszm/downscaling_1/zarr/test.zarr'),
    )
    print(dataset.__len__())
    print(dataset._static_vars())
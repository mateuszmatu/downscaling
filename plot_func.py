import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def plot_fields(ds, time_index: int, vars=['abs_vel', 'u_eastward', 'v_northward']) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(
        nrows=len(vars),
        ncols=4,
        figsize=(20, 4 * len(vars)),
        constrained_layout=True,
    )
    ds = xr.open_dataset(ds).isel(time=time_index)

    if len(vars) == 1:
        axes = np.array([axes])

    for i, var in enumerate(vars):
        if var == 'abs_vel':
            pfield = np.sqrt(ds['predicted_u_eastward'].values**2 + ds['predicted_v_northward'].values**2)
            ifield = np.sqrt(ds['input_u_eastward'].values**2 + ds['input_v_northward'].values**2)
            i800 = np.sqrt(ds['coarse_u_eastward'].values**2 + ds['coarse_v_northward'].values**2)
        else:
            pfield = ds[f'predicted_{var}'].values
            ifield = ds[f'input_{var}'].values
            i800 = ds[f'coarse_{var}'].values

        vmin = np.nanmin(ifield)
        vmax = np.nanmax(ifield)
        diff_limit = max(abs(vmin)/2, abs(vmax)/2)
        ax_coarse, ax_pred, ax_input, ax_diff = axes[i]

        im_coarse = ax_coarse.imshow(i800, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax_coarse.set_title(f"Input {var}")
        fig.colorbar(im_coarse, ax=ax_coarse, fraction=0.046, pad=0.04)

        im_pred = ax_pred.imshow(pfield, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"Downscaled {var}")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        im_input = ax_input.imshow(ifield, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax_input.set_title(f"Truth {var}")
        fig.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)

        im_diff = ax_diff.imshow(pfield - ifield, cmap='bwr', origin='lower', vmin=-diff_limit, vmax=diff_limit)
        ax_diff.set_title(f"Downscaled - Truth {var}")
        fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

    fig.savefig('results/predicted_fields.png', dpi=150)
    plt.close(fig)


def area_mean_timeseries(ds, vars=['abs_vel', 'u_eastward', 'v_northward']) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(
        nrows=len(vars),
        ncols=1,
        figsize=(12, 3.5 * len(vars)),
        constrained_layout=True,
    )
    ds = xr.open_dataset(ds)
    times = ds['time'].values

    if len(vars) == 1:
        axes = np.array([axes])

    for i, var in enumerate(vars):
        if var == 'abs_vel':
            pred = np.sqrt(ds['predicted_u_eastward'].values**2 + ds['predicted_v_northward'].values**2)
            truth = np.sqrt(ds['input_u_eastward'].values**2 + ds['input_v_northward'].values**2)
        else:
            pred = ds[f'predicted_{var}'].values
            truth = ds[f'input_{var}'].values

        diff = pred - truth

        ax = axes[i]
        ax.plot(times, np.nanmean(truth, axis=(1, 2)), label='Truth Area Mean', linewidth=2)
        ax.plot(times, np.nanmean(pred, axis=(1, 2)), label='Downscaled Area Mean', linewidth=2)
        ax.plot(times, np.nanmean(diff, axis=(1, 2)), label='Downscaled - Truth', linewidth=1.5, linestyle='--')
        ax.axhline(0.0, color='black', linewidth=1, alpha=0.5)
        ax.set_title(f"Area Mean Timeseries {var}")
        ax.set_ylabel('Area mean')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    axes[-1].set_xlabel('Time')

    fig.savefig('results/area_mean_timeseries.png', dpi=150)
    plt.close(fig)
"""Glue the 160m NorFjords grid tiles into one 160m bathymetry covering the NorKyst 800m domain.

Where the 160m tiles do not cover the 800m domain, h is bilinearly interpolated from the 800m grid.
All grids share the same polar stereographic projection (only false easting/northing differ),
so the tiles are placed on the target grid by projecting their lon_rho/lat_rho into the 800m
projection coordinates.
"""
import glob
import os

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

GRID_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_800 = os.path.join(GRID_DIR, "norkyst_grd_v31.nc")
FILES_160 = sorted(glob.glob(os.path.join(GRID_DIR, "norfjords_160m_grid_part*.nc")))
OUT_FILE = os.path.join(GRID_DIR, "norkyst_160m_grd_glued.nc")
DX = 160.0
ROW_CHUNK = 256


def main():
    g800 = xr.open_dataset(FILE_800)
    crs = CRS.from_proj4(g800["grid_mapping"].attrs["proj4string"])
    to_xy = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    to_ll = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

    x800 = g800["xi_rho"].values
    y800 = g800["eta_rho"].values
    h800 = g800["h"].values

    # Sanity check: projecting the 800m lon/lat must reproduce its own xi_rho/eta_rho coordinates
    xc, yc = to_xy.transform(g800["lon_rho"].values[::50, ::50], g800["lat_rho"].values[::50, ::50])
    err = max(np.abs(xc - x800[None, ::50]).max(), np.abs(yc - y800[::50, None]).max())
    print(f"800m projection check: max error {err:.3f} m")
    assert err < 1.0

    # Target 160m grid spanning the full 800m domain
    x160 = np.arange(x800[0], x800[-1] + DX / 2, DX)
    y160 = np.arange(y800[0], y800[-1] + DX / 2, DX)
    ny, nx = y160.size, x160.size
    print(f"Target 160m grid: eta_rho={ny}, xi_rho={nx}")

    h = np.full((ny, nx), np.nan)
    source = np.zeros((ny, nx), dtype=np.int8)

    for k, f in enumerate(FILES_160, start=1):
        ds = xr.open_dataset(f)
        lon = ds["lon_rho"].values
        lat = ds["lat_rho"].values
        hp = ds["h"].values
        ds.close()

        xp, yp = to_xy.transform(lon, lat)
        i0 = int(np.rint((xp[0, 0] - x160[0]) / DX))
        j0 = int(np.rint((yp[0, 0] - y160[0]) / DX))
        ni, nj = hp.shape[1], hp.shape[0]

        # Verify the tile is a regular 160m grid aligned with the target grid
        xi_exp = x160[0] + (i0 + np.arange(ni)) * DX
        yj_exp = y160[0] + (j0 + np.arange(nj)) * DX
        err = max(np.abs(xp - xi_exp[None, :]).max(), np.abs(yp - yj_exp[:, None]).max())
        print(f"{os.path.basename(f)}: offset (eta={j0}, xi={i0}), shape {hp.shape}, "
              f"alignment error {err:.3f} m")
        assert err < 1.0, f"{f} is not aligned with the target 160m grid"

        # Clip tile to the target domain
        tj0, tj1 = max(j0, 0), min(j0 + nj, ny)
        ti0, ti1 = max(i0, 0), min(i0 + ni, nx)
        if tj0 >= tj1 or ti0 >= ti1:
            print("  -> outside target domain, skipped")
            continue
        sub = hp[tj0 - j0:tj1 - j0, ti0 - i0:ti1 - i0]

        target = h[tj0:tj1, ti0:ti1]
        overlap = ~np.isnan(target)
        if overlap.any():
            print(f"  overlap with previous tiles: {overlap.sum()} points, "
                  f"max |dh| = {np.abs(target[overlap] - sub[overlap]).max():.3e} m")
        h[tj0:tj1, ti0:ti1] = sub
        source[tj0:tj1, ti0:ti1] = k

    missing = np.isnan(h)
    print(f"Points covered by 160m tiles: {100 * (1 - missing.mean()):.2f} %")

    # Fill gaps by bilinear interpolation of 800m h, and compute lon/lat of the target grid
    interp = RegularGridInterpolator((y800, x800), h800, method="linear")
    lon_out = np.empty((ny, nx))
    lat_out = np.empty((ny, nx))
    for j in range(0, ny, ROW_CHUNK):
        jj = slice(j, min(j + ROW_CHUNK, ny))
        X, Y = np.meshgrid(x160, y160[jj])
        lon_out[jj], lat_out[jj] = to_ll.transform(X, Y)
        m = missing[jj]
        if m.any():
            h[jj][m] = interp(np.column_stack([Y[m], X[m]]))

    # Check lon/lat against the 800m grid at coincident points
    err = max(np.abs(lon_out[::5, ::5] - g800["lon_rho"].values).max(),
              np.abs(lat_out[::5, ::5] - g800["lat_rho"].values).max())
    print(f"lon/lat check vs 800m grid: max error {err:.2e} deg")

    out = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), h, {
                "long_name": "Final bathymetry at RHO-points",
                "standard_name": "sea_floor_depth",
                "units": "meter",
                "coordinates": "lon_rho lat_rho",
                "grid_mapping": "grid_mapping",
            }),
            "h_source": (("eta_rho", "xi_rho"), source, {
                "long_name": "source of h",
                "flag_values": np.arange(len(FILES_160) + 1, dtype=np.int8),
                "flag_meanings": "norkyst800_interpolated " + " ".join(
                    os.path.splitext(os.path.basename(f))[0] for f in FILES_160),
            }),
            "lon_rho": (("eta_rho", "xi_rho"), lon_out, g800["lon_rho"].attrs),
            "lat_rho": (("eta_rho", "xi_rho"), lat_out, g800["lat_rho"].attrs),
            "grid_mapping": ((), np.int32(0), {**g800["grid_mapping"].attrs, "dx": np.int32(DX)}),
        },
        coords={
            "xi_rho": ("xi_rho", x160, g800["xi_rho"].attrs),
            "eta_rho": ("eta_rho", y160, g800["eta_rho"].attrs),
        },
        attrs={
            "title": "NorKyst 160m bathymetry glued from NorFjords 160m tiles, "
                     "gaps filled with bilinearly interpolated NorKyst 800m",
            "source_files": ", ".join(os.path.basename(f) for f in [FILE_800, *FILES_160]),
        },
    )
    g800.close()

    enc = {v: {"zlib": True, "complevel": 4} for v in ("h", "h_source", "lon_rho", "lat_rho")}
    out.to_netcdf(OUT_FILE, encoding=enc)
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()

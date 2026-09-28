import xarray as xr
import os


def cutout(path='/lustre/storeB/users/mateuszm/NF160/z/', domain='A01'):
    files = os.listdir(path+domain)


    regions = {
        'A01':
            {'A01_1': {'X': (10, 710), 'Y': (10, 710)},
            'A01_2': {'X': (780, 1480), 'Y': (10, 710)}},
        'A02':
            {'A02_1': {'X': (0, 700), 'Y': (10, 710)},
            'A02_2': {'X': (0, 700), 'Y': (750, 1450)}}
    }
    for file in files:
        print(file)
        ds = xr.open_dataset(path+domain+'/'+file).isel(depth=0)

        for region in regions[domain].keys():
            dsr = ds.isel(X=slice(regions[domain][region]['X'][0], regions[domain][region]['X'][1]), Y=slice(regions[domain][region]['Y'][0], regions[domain][region]['Y'][1]))

            if not os.path.exists('/lustre/storeB/users/mateuszm/NF160/cutout-regions/'+region):
                os.makedirs('/lustre/storeB/users/mateuszm/NF160/cutout-regions/'+region)
            dsr.to_netcdf('/lustre/storeB/users/mateuszm/NF160/cutout-regions/'+region+'/'+file+'_'+region)
        
cutout(domain='A01')
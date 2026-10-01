import xarray as xr
import os
import datetime

def cutout(start_date, end_date, path='/lustre/storeB/users/mateuszm/NF160/z/', domain='A01'):

    regions = {
        'A01':
            {'A01_1': {'X': (10, 710), 'Y': (10, 710)},
            'A01_2': {'X': (780, 1480), 'Y': (10, 710)}},
        'A02':
            {'A02_1': {'X': (0, 700), 'Y': (10, 710)},
            'A02_2': {'X': (0, 700), 'Y': (750, 1450)}}
    }

    delta = end_date - start_date
    for d in range(delta.days+1):
        now = start_date + datetime.timedelta(days=d)
        print(f'Now running {now}')
        later = now+datetime.timedelta(days=1)
        file = f'norfjords_160m_his.nc4_{now.year}{now.month:02d}{now.day:02d}01-{later.year}{later.month:02d}{later.day:02d}00_zdepth'
        ds = xr.open_dataset(path+domain+'/'+file).isel(depth=0)

        for region in regions[domain].keys():
            print('Cutting out region: '+region)
            dsr = ds.isel(X=slice(regions[domain][region]['X'][0], regions[domain][region]['X'][1]), Y=slice(regions[domain][region]['Y'][0], regions[domain][region]['Y'][1]))

            if not os.path.exists('/lustre/storeB/users/mateuszm/NF160/cutout-regions/'+region):
                os.makedirs('/lustre/storeB/users/mateuszm/NF160/cutout-regions/'+region)
            dsr.to_netcdf('/lustre/storeB/users/mateuszm/NF160/cutout-regions/'+region+'/'+file+'_'+region)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog = 'Transform',
        description='Transform Norfjord file to z depth'
    )
    parser.add_argument(
        '-s', '--time_start', default='2024-01-01'
    )
    parser.add_argument(
        '-e', '--time_stop', default='2024-01-01'
    )
    parser.add_argument(
        '-d', '--domain', default='A01'
    )

    args = parser.parse_args()


    start_date = datetime.datetime.strptime(args.time_start, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(args.time_stop,'%Y-%m-%d').date()
    cutout(start_date=start_date, end_date=end_date, domain=args.domain)

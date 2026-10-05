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
            'A02_2': {'X': (0, 700), 'Y': (750, 1450)}},
        'A03':
            {'A03_1': {'X': (450, 1150), 'Y': (10, 710)}},
        'A04':
            {'A04_1': {'X': (300, 1000), 'Y': (200, 900)}},
        'A05':
            {'A05_1': {'X': (10, 710), 'Y': (10, 710)},
            'A05_2': {'X': (10, 710), 'Y': (560, 1260)},
            'A05_3': {'X': (740, 1440), 'Y': (560, 1260)},
            'A05_4': {'X': (740, 1440), 'Y': (10, 710)}},
        'A06':
            {'A06_1': {'X': (10, 710), 'Y': (200, 900)}},
        'A07':
            {'A07_1': {'X': (10, 710), 'Y': (400, 1100)},
             'A07_2': {'X': (750, 1450), 'Y': (10, 710)},
             'A07_3': {'X': (800, 1500), 'Y': (460, 1160)}},
        'A08':
            {'A08_1': {'X': (10, 710), 'Y': (100, 800)}},
        'A09':
            {'A09_1': {'X': (10, 710), 'Y': (10, 710)},
             'A09_2': {'X': (700, 1400), 'Y': (100, 800)}},
        'A10':
            {'A10_1': {'X': (50, 750), 'Y': (10, 710)},
             'A10_2': {'X': (750, 1450), 'Y': (10, 710)},
             'A10_3': {'X': (1290, 1990), 'Y': (10, 710)},
             'A10_4': {'X': (300, 1000), 'Y': (440, 1140)},
             'A10_5': {'X': (1000, 1700), 'Y': (440, 1140)}},
        'A11':
            {'A11_1': {'X': (10, 710), 'Y': (500, 1200)},
             'A11_2': {'X': (700, 1400), 'Y': (400, 1100)},
             'A11_3': {'X': (1290, 1990), 'Y': (590, 1290)},
             'A11_4': {'X': (1290, 1990), 'Y': (710, 1290)}},
        'A12':
            {'A12_1': {'X': (10, 710), 'Y': (500, 1200)},
             'A12_2': {'X': (700, 1400), 'Y': (370, 1070)},
             'A12_3': {'X': (1340, 2040), 'Y': (10, 710)}},
        'A13':
            {'A13_1': {'X': (90, 790), 'Y': (70, 770)}},
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

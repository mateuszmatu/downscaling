import fimex
import os
from fix_u_and_v import fix_u_and_v
import time
import datetime
import xarray as xr
import numpy as np
import netCDF4


def transform(start_date, end_date, domain):
    abs_path = '/home/mateuszm/downscaling/preprocess_norfjord/'
    delta = end_date - start_date
    for d in range(delta.days+1):
        now = start_date + datetime.timedelta(days=d)
        print(f'Now running {now}')
        later = now+datetime.timedelta(days=1)
        path1 = f'/lustre/storeB/users/mateuszm/NF160/s/{domain}/'
        ifile = f'norfjords_160m_his.nc4_{now.year}{now.month:02d}{now.day:02d}01-{later.year}{later.month:02d}{later.day:02d}00'

        ## add s_w (it was missing from the data)
        layers = np.linspace(-1,0,36)
        with netCDF4.Dataset(path1+ifile, mode='a') as ncfile:
            try:
                s_w = ncfile.createVariable('s_w', np.double,('s_w'))
                s_w[:] = layers
                s_w.long_name = "S-coordinate at W-points"
                s_w.valid_min = -1
                s_w.valid_max = 0
                s_w.positive = "up"
                s_w.standard_name = "ocean_s_coordinate_g2"
                s_w.formula_terms = "s: s_w C: Cs_w eta: zeta depth: h depth_c: hc"
                s_w.field = "s_w, scalar"
            except: 
                pass

        destagg = abs_path + f'destagg_{domain}.nc'
        destagg_config_file = abs_path + 'input/norfjord_OutputConfig_ROMS.xml'
        input_config = abs_path + f'input/Norfjords_{domain}_vertint_input.ncml'
        config_file = abs_path + f'input/norfjord_destagg_{domain}.cfg'

        #destagger
        command = 'fimex -n '+'8'+ ' -c '+config_file+ ' --input.config='+input_config+' --input.file='+path1+ifile+' --output.file='+destagg+' --output.config='+destagg_config_file
        os.system(command)

        #fix u and v
        fix_u_and_v(destagg)

        #rotate u and v
        rotate = abs_path+f'rotated_{domain}.nc'
        o = open(f"/home/mateuszm/downscaling/preprocess_norfjord/fimex_N160_{domain}.cfg","w")
        incfg = abs_path+'fimex/fimex_rot.cfg'
        for line in open(incfg):
            line = line.replace('INPUT', destagg)
            line = line.replace('OUTPUT', rotate)
            o.write(line + "\n")
        o.close()
        if not os.system(f"fimex -c /home/mateuszm/downscaling/preprocess_norfjord/fimex_N160_{domain}.cfg") == 0:
            print('Problems rotating U and V')

        os.system(f'ncrename -v u,u_eastward -v v,v_northward {rotate}')

        path2 = f'/lustre/storeB/users/mateuszm/NF160/z/{domain}/'
        ofile = ifile+'_zdepth'
        config_file = abs_path+'input/norfjord_vertint.cfg'
        
        #Vertical interpolation
        cfg = fimex.FimexConfig()
        cfg.read_cfg(config_file)
        cfg.addattr('ncml', 'config', abs_path+'input/norfjord_vertint_names.ncml')
        cfg.addattr('input', 'file', rotate)
        cfg.addattr('output', 'file', path2+ofile)
        cfg.run_fimex()

        #Clean up temporary files
    os.system(f'rm {destagg}')
    os.system(f'rm {rotate}')

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
    transform(start_date, end_date, args.domain)

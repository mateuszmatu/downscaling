# Collection of tools used in ecflow for ocean and ice dept.
# nilsmk 2019
#
def fix_u_and_v(ifile):
	import netCDF4
	import numpy as np
	import time
	t0 = time.time()
	nc = netCDF4.Dataset(ifile,'r+')
	print('Infile: '+ifile)
	temp = nc.variables['temp']
	u    = nc.variables['u']
	v    = nc.variables['v']
	#ue   = nc.variables['u_eastward']
	#vn   = nc.variables['v_northward']
	tempmask = temp[0,0,:].mask.copy()
	umask    = u[0,0,:].mask.copy()
	vmask    = v[0,0,:].mask.copy()
	tu_mask  = (umask != tempmask)
	tv_mask  = (vmask != tempmask)
	for i in range(temp.shape[0]):
		print('----------------------------')
		print('timesteps left: {}'.format(temp.shape[0]-i))
		t1 = time.time()
		u[i,:][:,tu_mask] = 0.
		v[i,:][:,tv_mask] = 0.
		#ue[i,:][:,tu_mask] = 0.
		#vn[i,:][:,tv_mask] = 0.
		t2 = time.time()
		nc.sync()
		print('sync took {:.4f}'.format(time.time()-t2))
		print('took {:.1f} seconds'.format(time.time()-t1))
		print('estimate {:.1f} min left'.format((temp.shape[0]-i-1)*(time.time()-t1)*(1./60.)))
		print('so far {:.1f} min total'.format((time.time()-t0)*(1./60.)))
	nc.sync()
	nc.close()
	print('----------------------------')
	print('== All took {} minutes =='.format((time.time()-t0)*(1./60.)))

def blend_climfiles(file1, file2, no_left=0, no_bott=0, x1=0, x2=0, y1=0, y2=0, blending_width =0, blendvar='all', method='old'):
	# func will blend file 1 onto file 2
	# no_left is how many gridpoints along left side of domain to replace
	# no_bott is how many gridpoints along lower side of domain to replace
	# blending_width is the number of points to blend the two products over, not simply replace one by the other.
	# To ensure a smooth transition for use in initial condtions etc.

	import numpy as np
	import netCDF4
	import sys
	from scipy import ndimage
	#
	if blendvar == 'all':
		variables = nc1.variables
	else:
		variables = blendvar
	nc1 = netCDF4.Dataset(file1)
	nc2 = netCDF4.Dataset(file2, 'r+')
	if (len(nc1.variables['clim_time'][:]) != 1) and (len(nc1.variables['clim_time'][:]) != len(nc2.variables['clim_time'][:])):
		print('only works for clim_time length 1 or equal between files...')
		sys.exit(1)
	for var in variables:
		if method == 'new':
			if not np.array_equal(nc1.variables[var].shape[-2:], nc2.variables[var].shape[-2:]):
				print('X- and y-dims on input files must be equal!!')
				sys.exit(1)
			if blending_width:
				weights = np.ones([y2-y1, x2-x1])
				if x1 != 0:
				    weights[:,0] = 0
				if y1 != 0:
				    weights[0,:] = 0
				if x2  != nc1.variables[var].shape[-1]:
				    weights[:,-1] = 0
				if y2  != nc1.variables[var].shape[-2]:
				    weights[-1,:] = 0
				# expand boundary zone:
				weights = ndimage.uniform_filter(weights, output = np.zeros(weights.shape), size=(blending_width, blending_width), mode = 'nearest')
				# rescale weights to fall between 0 and 1
				weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

			else:
				weights = np.ones([y2-y1, x2-x1])
		if len(nc1.variables[var].shape) == 4:
			print(len(nc1.variables[var].shape), var)
			weights = np.repeat(weights[:, :, np.newaxis], nc2.variables[var].shape[1], axis=2)
			weights = np.moveaxis(weights, -1, 0)
			for t in range(nc2.variables[var].shape[0]):
				if len(nc1.variables['clim_time'][:]) == 1:
					t2 = 0
				else:
					t2 = t
					print('assume time on both files are the same!!')
				if method == 'old':
					nc2.variables[var][t,:,0:no_bott,0:nc1.variables[var].shape[-1]] = nc1.variables[var][t2,:,0:no_bott,:]
					nc2.variables[var][t,:,0:nc1.variables[var].shape[-2],0:no_left] = nc1.variables[var][t2,:,:,0:no_left]
				elif method == 'new':
					nc2.variables[var][t, :, y1:y2 , x1:x2] = nc1.variables[var][t2, :, y1:y2, x1:x2] * weights[:, :, :] + nc2.variables[var][t, :, y1:y2 , x1:x2]*(1-weights[:, :, :])
				else:
					print('Unknown method!!')
					sys.exit(1)
		elif len(nc1.variables[var].shape) == 3:
			print(len(nc1.variables[var].shape), var)
			for t in range(nc2.variables[var].shape[0]):
				if len(nc1.variables['clim_time'][:]) == 1:
					t2 = 0
				else:
					t2 = t
					print('assume time on both files are the same!!')
				if method == 'old':
					nc2.variables[var][t,0:no_bott,0:nc1.variables[var].shape[-1]] = nc1.variables[var][t2,0:no_bott,:]
					nc2.variables[var][t,0:nc1.variables[var].shape[-2],0:no_left] = nc1.variables[var][t2,:,0:no_left]
				elif method == 'new':
					nc2.variables[var][t, y1:y2 , x1:x2] = nc1.variables[var][t2, y1:y2, x1:x2] * weights[0, :, :] + nc2.variables[var][t,  y1:y2 , x1:x2]*(1-weights[0, :, :])

				else:
					print('Unknown method!!')
					sys.exit(1)
		else:
			print('i wont touch '+var)
		nc2.sync()
	nc2.close()

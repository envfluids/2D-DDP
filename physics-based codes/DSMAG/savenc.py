import numpy as np
import netCDF4 as nc4
import os

def savenc(psi_in,w_in, lon,lat, t, filename):
    directory = filename
    os.makedirs(directory)
    f = nc4.Dataset(filename + '/' + filename + '.nc','w', format='NETCDF4')
    tempgrp = f.createGroup('Temp_data')
    tempgrp.createDimension('lon', len(lon))
    tempgrp.createDimension('lat', len(lat))
    tempgrp.createDimension('t', len(t))
    
    longitude = tempgrp.createVariable('Longitude', 'f4', 'lon')
    latitude = tempgrp.createVariable('Latitude', 'f4', 'lat')  
    psi = tempgrp.createVariable('PSI', 'f4', ('lat','lon','t'))
    omega = tempgrp.createVariable('OMEGA', 'f4', ('lat','lon','t'))
    time = tempgrp.createVariable('Time', 'f4', 't')

    longitude[:] = lon
    latitude[:] = lat
    time[:] = t
	
    print(np.shape(psi))
    print(np.shape(psi_in))
	

    psi[:,:,:] = psi_in
    omega[:,:,:] = w_in
  
    f.close()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:20:05 2022

@author: skrunes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:31:45 2022

@author: skrunes
"""

import numpy as np
#import numpy.linalg as npl
import netCDF4 as nc
import xarray as xr
#import global_diffusion.diffusion_operators as diff
#import global_diffusion.diffusion_normalization as norm
#import global_diffusion.diffusion_auxiliary as aux
#import time
import sys
import scipy.ndimage.morphology as scm
from scipy import interpolate
# Packages required only for plot_anomaly
#import matplotlib.pyplot as plt
#import mpl_toolkits.basemap as mpb
import os
import os.path
import datetime
from matplotlib import pyplot as plt

def pad_lon(X,pad):
    
    padded = np.empty((X.shape[0],X.shape[1] + 2 * pad))
    padded[:,pad:-pad] = np.copy(X)
    padded[:,:pad] = np.copy(padded[:,-2*pad:-pad])
    padded[:,-pad:] = np.copy(padded[:,pad:2*pad])
    
    return padded

def pad_lat(X,pad,roll=False,remove_copy=None):
    
    padded = np.ma.zeros((X.shape[0] + 2 * pad,X.shape[1] ))
    padded[pad:-pad,:] = np.copy(X)
    
    if remove_copy is not None :
    
        roll1 = np.copy(padded[-2*pad-remove_copy:-pad-remove_copy,X.shape[1]//2:]) 
        roll1 = np.flip(roll1,axis=(0,1))
    
    
        roll2 = np.copy(padded[-2*pad-remove_copy:-pad-remove_copy,:X.shape[1]//2])
        roll2 = np.flip(roll2,axis=(0,1))
    else:
        roll1 = np.copy(padded[-2*pad:-pad,X.shape[1]//2:]) 
        roll1 = np.flip(roll1,axis=(0,1))
    
    
        roll2 = np.copy(padded[-2*pad:-pad,:X.shape[1]//2])
        roll2 = np.flip(roll2,axis=(0,1))
    if roll is not None :
        roll1 = np.roll(roll1,roll)
        roll2 = np.roll(roll2,roll)
    
        
    roll3 = np.copy(padded[pad:2*pad,X.shape[1]//2:])
    roll3 = np.flip(roll3,axis=(0,1))
    
    roll4 = np.copy(padded[pad:2*pad,:X.shape[1]//2])
    roll4 = np.flip(roll4,axis=(0,1))

        
    padded[-pad:,:X.shape[1]//2] = roll1
    padded[-pad:,X.shape[1]//2:] = roll2
    
    padded[:pad,:X.shape[1]//2] = roll3
    padded[:pad,X.shape[1]//2:] = roll4
    
    #padded[pad-1,:] = np.zeros(360)
    
    
    return padded

with nc.Dataset("../data/coordinates.nc", 'r') as data:
    e1u = data["e1u"][:]
    e2u = data["e2u"][:]
    
    e1t = data["e1t"][:]
    e2t = data["e2t"][:]
    
    e1f = data["e1f"][:]
    e2f = data["e2f"][:]
    
    e1v = data["e1v"][:]
    e2v = data["e2v"][:]
    
    gphiu = data["gphiu"][:]
    gphif = data["gphif"][:]
    gphit = data["gphit"][:]
    gphiv = data["gphiv"][:]
    
    glamu = data["glamu"][:]
    glamf = data["glamf"][:]
    glamt = data["glamt"][:]
    glamv = data["glamv"][:]
    
data_path = "../data/NORSSH_BND_PERTEN_RESULTS/"
for file_list in os.listdir(data_path):
    
    if file_list!=".DS_Store" and file_list!="rename_res.py":
        
            
        with nc.Dataset(data_path + file_list+"/Bmod_dif_per_mod1_wri_ssh.nc", 'r') as data:
            K11 = data["K11"][:]
            K22 = data["K22"][:]
            #nav_lon = data["nav_lon"][:]
            #nav_lat = data["nav_lat"][:]
        
        with nc.Dataset(data_path + file_list+"/Bmod_cor_nor_per_mod1_wri_ssh.nc", 'r') as data:
            gamma = data["sqtnorfac"][:]   
            
    else:
        pass
    # alpha_i =  np.nan_to_num(K22[0,1:-1,1:-1] * e1f[1:-1,1:-1] / e2t[1:-1,1:-1])
    # #alpha_i_minus = kappa_i[:-1, :] * ej_parallel[:-1, :] / ei_parallel[:-1, :] 

    # #alpha_j_plus = kappa_j[:, 1:] * ei_meridian[:, 1:] / ej_meridian[:, 1:] 
    # alpha_j = np.nan_to_num(K11[0,1:-1,1:-1] * e2f[1:-1,1:-1] / e1t[1:-1,1:-1])
    # norm_coeffs = gamma[0,1:-1,1:-1]
    # mask_gamma=np.ma.getmask(norm_coeffs)
    # mask_K11=np.ma.getmask(K11[0,1:-1,1:-1])
    # mask_K22=np.ma.getmask(K22[0,1:-1,1:-1])
    # alpha_i[mask_K22] = 0
    # alpha_j[mask_K11] = 0
    # norm_coeffs[mask_gamma] = 0
   

    npad_lat=31
    npad_lon = 28
     

    mask=np.ma.getmask(gamma)
    mask_pad = pad_lat(mask[0,1:-1,1:-1],npad_lat,remove_copy=None)
    mask_pad = np.array(pad_lon(mask_pad,npad_lon),dtype=bool)
    
    mask_K22=np.ma.getmask(K22)
    mask_K22_pad = pad_lat(mask_K22[0,:,1:-1],npad_lat,remove_copy=3)
    mask_K22_pad = np.array(pad_lon(mask_K22_pad,npad_lon),dtype=bool)
    
    mask_K11=np.ma.getmask(K11)
    mask_K11_pad = pad_lat(mask_K11[0,1:-1,1:-1],npad_lat,roll=None,remove_copy=None)
    mask_K11_pad = np.array(pad_lon(mask_K11_pad,npad_lon),dtype=bool)
    
    
    K22_pad = pad_lat(K22[0,:,1:-1],npad_lat,remove_copy=3)
    K22_pad = pad_lon(K22_pad,npad_lon)
    #K22_pad = np.ma.masked_array(K22_pad, mask_K22_pad)
    K22_pad[mask_K22_pad] = 0
    
    K11_pad = pad_lat(K11[0,1:-1,1:-1],npad_lat,roll=None,remove_copy=None)
    K11_pad = pad_lon(K11_pad,npad_lon)
    #K11_pad = np.ma.masked_array(K11_pad, mask_K11_pad)
    K11_pad[mask_K11_pad] = 0
    
    gamma_pad = pad_lat(gamma[0,1:-1,1:-1],npad_lat,remove_copy=None)
    gamma_pad = pad_lon(gamma_pad,npad_lon)
    #gamma_pad = np.ma.masked_array(gamma_pad, mask_pad)
    gamma_pad[mask_pad] = 0
    
    e1f_pad = pad_lat(e1f[1:-1,1:-1],npad_lat,roll=-1,remove_copy=1)
    e1f_pad = pad_lon(e1f_pad,npad_lon)

    
    e2t_pad = pad_lat(e2t[1:-1,1:-1],npad_lat)
    e2t_pad = pad_lon(e2t_pad,npad_lon)


    e1t_pad = pad_lat(e1t[1:-1,1:-1],npad_lat)
    e1t_pad = pad_lon(e1t_pad,npad_lon)

    
    e2f_pad = pad_lat(e2f[1:-1,1:-1],npad_lat,roll=-1,remove_copy=1)
    e2f_pad = pad_lon(e2f_pad,npad_lon)


    e1u_pad = pad_lat(e1u[1:-1,1:-1],npad_lat,roll=-1)
    e1u_pad = pad_lon(e1u_pad,npad_lon)
    #e1t_pad = np.ma.masked_array(e1t_pad, mask_pad)

    e2v_pad = pad_lat(e2v[1:-1,1:-1],npad_lat)#,roll=False,remove_copy=1)
    e2v_pad = pad_lon(e2v_pad,npad_lon)
    #e2f_pad = np.ma.masked_array(e2f_pad, mask_pad)
    # print(alpha_i_pad.shape)

    w_pad = e1u_pad*e2v_pad
    
    alpha_i_pad = K22_pad[1:-1,:] * e1f_pad / e2t_pad
    alpha_j_pad = K11_pad * e2f_pad / e1t_pad


    
    
    
    
    
    
    
    
    lats = np.arange(len(alpha_i_pad),dtype=float)
    lons = np.arange(len(alpha_i_pad[0]),dtype=float)


    data_vars = {"alpha_i_minus":(["lat","lon"],alpha_i_pad),
                      "alpha_j_minus":(["lat","lon"],alpha_j_pad),
                      "w":(["lat","lon"],w_pad),
                      "norm_coeffs":(["lat","lon"],gamma_pad),
                      
                  }
    coords ={"lat": (["lat"], lats),"lon":(["lon"],lons)}

                         
    #file_name = "data/hetero_edge/id_" + str(seed) + ".nc"
    file_name = "../data/raw/nemo_bnd_perten/"+file_list+".nc"
    ds = xr.Dataset(data_vars=data_vars, 
                coords=coords)
    ds.to_netcdf(file_name)


    








#v = print(nav_lat[:,0])

# def d_lat_plus(x,d):
#     """
    

#     Parameters
#     ----------
#     x : tuple
#         (lat,lon)
#     d : float, optional
#         DESCRIPTION. The default is 1.

#     Returns
#     -------
#     y : new point (lat,lon)

#     """
#     lat,lon = x
#     dlat,dlon = d
#     assert lat+dlat < 90, "Noooo"
#     assert abs(lon + dlon) -180 != 0, "Ayyyy" 
 
#     if lon + dlon > 180:
                                                                                                                                                                                                          
#         lon = -180 + (lon + dlon - 180)
#     elif lon + dlon < -180:
#         lon = 180 - abs(lon + dlon + 180)
    
#     return (lat + dlat, lon)

# def norm(x,y_lat,y_lon):
#     """
    

#     Parameters
#     ----------
#     x : tuple ()
#         DESCRIPTION.
#     y : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
    
#     aux1 = (x[0] - y_lat)**2
#     aux2 = (x[1] - y_lon)**2
    
#     return(np.sqrt(aux1+aux2))

data_path = "../data/NORSSH_PERTEN_RESULTS/"
with nc.Dataset(data_path +"NORSSH_PERTEN30"+"/Bmod_dif_per_mod1_wri_ssh.nc", 'r') as data:
    K11 = data["K11"][:]
    K22 = data["K22"][:]
    nav_lon = data["nav_lon"][:]
    nav_lat = data["nav_lat"][:]
with nc.Dataset(data_path +"NORSSH_PERTEN30"+"/Bmod_cor_nor_per_mod1_wri_ssh.nc", 'r') as data:
    
    gamma = data["sqtnorfac"][:]   



# # for i in range(len(nav_lat[-1,:])):
# #     x = d_lat_plus((nav_lat[-1,i],nav_lon[-1,i]),(d_lat[-1,i],d_lon[-1,i]))
# #     print(np.unravel_index(np.argmin(norm(x,nav_lat[:,:],nav_lon[:,:])),shape=nav_lat.shape))

# #Conversion to alpha
# #alpha_i =  np.ma.masked_array(K22 * e1f / e2t  , mask)
# # alpha_i_minus = kappa_i[:-1, :] * ej_parallel[:-1, :] / ei_parallel[:-1, :] 

# # alpha_j_plus = kappa_j[:, 1:] * ei_meridian[:, 1:] / ej_meridian[:, 1:] 
# #alpha_j =  np.ma.masked_array(K11 * e2f / e1t  , mask)

# def getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2):
#   R = 6371
#   dLat = deg2rad(lat2-lat1)
#   dLon = deg2rad(lon2-lon1) 
#   a =  np.sin(dLat/2) * np.sin(dLat/2) + np.cos(deg2rad(lat1)) * np.cos(deg2rad(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
     
#   c = 2 * np.atan2(np.sqrt(a), np.sqrt(1-a)) 
#   d = R * c 
#   return d


# def deg2rad(deg):
#   return deg * (np.pi/180)
 


# w = e2u[:,1:-1]*e1v[:,1:-1]     

# npad_lat=100
# npad_lon = 100
     

# mask=np.ma.getmask(gamma)
# mask_pad = pad_lat(mask[0,1:-1,1:-1],npad_lat,remove_copy=None)
# mask_pad = pad_lon(mask_pad,npad_lon)

# mask_K22=np.ma.getmask(K22)
# mask_K22_pad = pad_lat(mask_K22[0,:,1:-1],npad_lat,remove_copy=3)
# mask_K22_pad = pad_lon(mask_K22_pad,npad_lon)

# mask_K11=np.ma.getmask(K11)
# mask_K11_pad = pad_lat(mask_K11[0,1:-1,1:-1],npad_lat,roll=None,remove_copy=None)
# mask_K11_pad = pad_lon(mask_K11_pad,npad_lon)


# # alpha_i_pad = pad_lat(K22[0,:,1:-1] * e1f[:,1:-1] / e2t[:,1:-1],npad_lat)
# # alpha_i_pad = pad_lon(alpha_i_pad,npad_lon)
# # alpha_i_pad = np.ma.masked_array(alpha_i_pad, mask_K22_pad)

# # alpha_j_pad = pad_lat(K11[0,:,1:-1] * e2f[:,1:-1] / e1t[:,1:-1],npad_lat)
# # alpha_j_pad = pad_lon(alpha_j_pad,npad_lon)
# # alpha_j_pad = np.ma.masked_array(alpha_j_pad, mask_K11_pad)

# K22_pad = pad_lat(K22[0,:,1:-1],npad_lat,remove_copy=3)
# K22_pad = pad_lon(K22_pad,npad_lon)
# K22_pad = np.ma.masked_array(K22_pad, mask_K22_pad)

# #K22_pad = K22[0,:,1:-1]

# K11_pad = pad_lat(K11[0,1:-1,1:-1],npad_lat,roll=None,remove_copy=None)
# K11_pad = pad_lon(K11_pad,npad_lon)
# K11_pad = np.ma.masked_array(K11_pad, mask_K11_pad)

# #K11_pad = K11[0,:-1,1:-1]

# #w_pad = pad_lat(w,npad_lat)
# #w_pad = pad_lon(w_pad,npad_lon)
# #w_pad = np.ma.masked_array(w_pad, mask_pad)

# nav_lon_pad = pad_lat(abs(nav_lon[:,1:-1])-180,npad_lat)
# nav_lon_pad = pad_lon(nav_lon_pad,npad_lon)
# #nav_lon_pad = np.ma.masked_array(nav_lon_pad, mask_pad)

# nav_lat_pad = pad_lat(nav_lat[:,1:-1],npad_lat)
# nav_lat_pad = pad_lon(nav_lat_pad,npad_lon)
# #nav_lat_pad = np.ma.masked_array(nav_lat_pad, mask_pad)

# gamma_pad = pad_lat(gamma[0,1:-1,1:-1],npad_lat,remove_copy=None)
# gamma_pad = pad_lon(gamma_pad,npad_lon)
# gamma_pad = np.ma.masked_array(gamma_pad, mask_pad)

# #gamma_pad = gamma[0,:-1,1:-1]

# e1f_pad = pad_lat(e1f[1:-1,1:-1],npad_lat,roll=-1,remove_copy=1)
# e1f_pad = pad_lon(e1f_pad,npad_lon)
# #e1f_pad = np.ma.masked_array(e1f_pad, mask_pad)

# e2t_pad = pad_lat(e2t[1:-1,1:-1],npad_lat)
# e2t_pad = pad_lon(e2t_pad,npad_lon)
# #e2t_pad = np.ma.masked_array(e2t_pad, mask_pad)


# e1t_pad = pad_lat(e1t[1:-1,1:-1],npad_lat)
# e1t_pad = pad_lon(e1t_pad,npad_lon)
# #e1t_pad = np.ma.masked_array(e1t_pad, mask_pad)

# e2f_pad = pad_lat(e2f[1:-1,1:-1],npad_lat,roll=-1,remove_copy=1)
# e2f_pad = pad_lon(e2f_pad,npad_lon)
# #e2f_pad = np.ma.masked_array(e2f_pad, mask_pad)
# # print(alpha_i_pad.shape)


# e1u_pad = pad_lat(e1u[1:-1,1:-1],npad_lat,roll=-1)
# e1u_pad = pad_lon(e1u_pad,npad_lon)
# #e1t_pad = np.ma.masked_array(e1t_pad, mask_pad)

# e2v_pad = pad_lat(e2v[1:-1,1:-1],npad_lat)#,roll=False,remove_copy=1)
# e2v_pad = pad_lon(e2v_pad,npad_lon)
# #e2f_pad = np.ma.masked_array(e2f_pad, mask_pad)
# # print(alpha_i_pad.shape)

# w_pad = e1u_pad*e2v_pad

# alpha_i_pad = K22_pad[1:-1,:] * e1f_pad / e2t_pad
# alpha_j_pad = K11_pad * e2f_pad / e1t_pad
# # alpha_i_pad = pad_lon(alpha_i_pad,npad_lon)
# # alpha_i_pad = np.ma.masked_array(alpha_i_pad, mask_K22_pad)

# # alpha_j_pad = pad_lat(K11[0,:,1:-1] * e2f[:,1:-1] / e1t[:,1:-1],npad_lat)
# # alpha_j_pad = pad_lon(alpha_j_pad,npad_lon)
# # alpha_j_pad = np.ma.masked_array(alpha_j_pad, mask_K11_pad)

# # #d_lat=nav_lat[:,1:]-nav_lat[:-1,:]
# # #d_lon=nav_lon[:,1:]-nav_lon[:-1,:]

# #plt.imshow(alpha_i_pad,origin="lower",aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
# lats = np.arange(len(e1u_pad),dtype=float)
# lons = np.arange(len(e1u_pad[0]),dtype=float)
# data_vars = {"alpha_i_minus":(["lat","lon"],alpha_i_pad),
#                        "alpha_j_minus":(["lat","lon"],alpha_j_pad),
#                        "K11":(["lat","lon"],K11_pad),
#                       #"K22":(["lat","lon"],K22_pad),
#                       "w":(["lat","lon"],w_pad),
#             #          "e1f":(["lat","lon"],e1f_pad),
#             #          "e2f":(["lat","lon"],e2f_pad),
#             #          "e1t":(["lat","lon"],e1t_pad),
#             #          "e2t":(["lat","lon"],e2t_pad),
#                       "norm_coeffs":(["lat","lon"],gamma_pad),
#                       "e2v":(["lat","lon"],e2v_pad),
#                       "e1u":(["lat","lon"],e1u_pad),
#                   }
# coords ={"lat": (["lat"], lats),"lon":(["lon"],lons)}

                         
# #file_name = "data/hetero_edge/id_" + str(seed) + ".nc"
# file_name = "test_lat_lon5.nc"
# ds = xr.Dataset(data_vars=data_vars, 
#             coords=coords)
# ds.to_netcdf(file_name)






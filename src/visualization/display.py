#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:28:42 2021

@author: goux
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import cartopy.crs as ccrs
import copy

def plot_swath_sample(swath_sample, nadir_sample=None, cmap = None, norm = None,
                      levels = 0, variable = None, title = None, unit = 'm', marker = None,
                      log = False, figsize = (30, 10)):
    
    # Mask the gap in the swath
    masked_sample = mask_gap(swath_sample)
    
    # Scale for the unit
    if unit == 'cm':
        masked_sample*=100
    if unit == 'mm':
        masked_sample*=1000
    
    # compute the distances
    along_track_dist, across_track_dist = distances(*swath_sample.shape, corners = True)
   
    
    # Default settings if no cmap or norm are specified
    if nadir_sample is not None:
        vmin = min([np.min(masked_sample),np.min(nadir_sample)])
        vmax = max([np.max(masked_sample),np.max(nadir_sample)])
    else:
        vmin = np.min(masked_sample)
        vmax = np.max(masked_sample)
    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    
    # Initialize figure
    plt.figure(figsize=figsize)
    
    # Plot swath data
    c = plt.pcolormesh(along_track_dist,across_track_dist, masked_sample.T, 
                cmap = cmap, norm=norm)
    
    #Plot nadir data
    if nadir_sample is not None:
        along_track_centers, _ = distances(*swath_sample.shape, corners = False)
        plt.scatter(along_track_centers, np.zeros_like(along_track_centers), c=nadir_sample,
                    cmap = cmap, norm=norm, s=6)
    
    if marker is not None:
        plt.scatter(marker[0], marker[1], marker='+',s=45,c='k')
    
    cbar = plt.colorbar(c)
    if variable is None:
        variable = "error (" + unit + ")" 
    cbar.ax.set_ylabel(variable, fontsize = 14)
    if title is not None:
        plt.suptitle(title)
    plt.xlabel("along track distance(km)", fontsize = 13)
    plt.ylabel("across track distance(km)", fontsize = 13)
    plt.show()
    
    
def plot_swath_globe(swath_data, lon_swath, lat_swath,
                     nadir_data=None, lon_nadir=None, lat_nadir=None,
                     cmap = None, norm = None, levels = 0, variable = None,
                     title = None, unit = 'm', marker = None, log = False, 
                     figsize = (30, 10)):
    
    Nalong, Nacross = swath_data.shape
    # swath_masked = mask_gap(swath_data)
    # lon_masked = mask_gap(lon_swath, fill_mean = True)
    # lat_masked = mask_gap(lat_swath, fill_mean = True)
    
    # Scale for the unit
    if unit == 'cm':
        swath_data*=100
    if unit == 'mm':
        swath_data*=1000
    

    
    # Default settings if no cmap or norm are specified
    if nadir_data is not None:
        vmin = min([np.min(swath_data),np.min(nadir_data)])
        vmax = max([np.max(swath_data),np.max(nadir_data)])
    else:
        vmin = np.min(swath_data)
        vmax = np.max(swath_data)
    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    # Initialize figure
    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_subplot(1, 1, 1,\
                         projection=ccrs.PlateCarree(central_longitude=0))
    # Plot swath data
    
    c = plt.pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       swath_data[:,:Nacross//2],  cmap = cmap, norm=norm, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = plt.pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       swath_data[:,-Nacross//2:],  cmap = cmap, norm=norm, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    #c = plt.pcolormesh(lon_swath, lat_swath,
                       #swath_data,  cmap = cmap, norm=norm, 
                       #transform=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines()
    ax.set_global()
    # Plot nadir data
    if nadir_data is not None:
        plt.scatter(lon_nadir, lat_nadir, c=nadir_data,
                    cmap = cmap, norm=norm, s=1, transform=ccrs.PlateCarree(central_longitude=180))
    
    if marker is not None:
        plt.scatter(marker[0], marker[1], marker='+',s=45,c='k')
    
    cbar = plt.colorbar(c)
    if variable is None:
        variable = "error "#(" + unit + ")" 
    cbar.ax.set_ylabel(variable, fontsize = 14)
    if title is not None:
        plt.suptitle(title)
    plt.show()

def plot_diff_globe(diff1,diff2, lon_swath, lat_swath,
                     nadir_data=None, lon_nadir=None, lat_nadir=None,
                     cmap = None, norm = None, levels = 0, variable = None,
                     title = None,cmap_title=None, unit = 'm', marker = None, log = False, 
                     figsize = (30, 10),vmin=None,vmax=None):
    
    Nalong, Nacross = diff1.shape
    # swath_masked = mask_gap(swath_data)
    # lon_masked = mask_gap(lon_swath, fill_mean = True)
    # lat_masked = mask_gap(lat_swath, fill_mean = True)
    
    

    
    # Default settings if no cmap or norm are specified
    if vmin is None:
        vmin = min([np.min(diff1),np.min(diff2)])
    if vmax is None:
        vmax = max([np.max(diff1),np.max(diff2)])
    #

    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    # Initialize figure
    fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.PlateCarree()})
    #fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.Mollweide()})

    # Plot swath data
    
    c = axs[0].pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       diff1[:,:Nacross//2],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs[0].pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       diff1[:,-Nacross//2:],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    axs[0].set_title("Train",fontsize=18)
    axs[0].coastlines()
    axs[0].set_global()
    
    c = axs[1].pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       diff2[:,:Nacross//2],  cmap = cmap,  vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs[1].pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       diff2[:,-Nacross//2:],  cmap = cmap,  vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    axs[1].set_title("Validation",fontsize=18)
    axs[1].coastlines()
    axs[1].set_global()
    
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)

    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])

    # Draw the colorbar
    cbar=fig.colorbar(c, cax=cbar_ax,orientation='horizontal')
    if cmap_title is not None:
        cbar.set_label(cmap_title, fontsize = 16)


    if title is not None:
        plt.suptitle(title,fontsize=26)
    plt.show()
    
def plot_error_globe(error, lon_swath, lat_swath,
                     nadir_data=None, lon_nadir=None, lat_nadir=None,
                     cmap = None, norm = None, levels = 0, variable = None,
                     title = None,cmap_title=None, unit = 'm', marker = None, log = False, 
                     figsize = (30, 10),vmin=None,vmax=None):
    
    Nalong, Nacross = error.shape
    # swath_masked = mask_gap(swath_data)
    # lon_masked = mask_gap(lon_swath, fill_mean = True)
    # lat_masked = mask_gap(lat_swath, fill_mean = True)
    
    # Default settings if no cmap or norm are specified
    if vmin is None:
        vmin = np.min(error)
    if vmax is None:
        vmax = np.max(error)
    #

    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    # Initialize figure
    fig,axs = plt.subplots(figsize=figsize,ncols=1,subplot_kw={'projection': ccrs.PlateCarree()})
    #fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.Mollweide()})

    # Plot swath data
    
    c = axs.pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       error[:,:Nacross//2],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs.pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       error[:,-Nacross//2:],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    #axs.set_title("Train",fontsize=18)
    axs.coastlines()
    axs.set_global()
    
    
    
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)

    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.266, 0.2, 0.4673, 0.02])

    # Draw the colorbar
    cbar=fig.colorbar(c, cax=cbar_ax,orientation='horizontal')
    if cmap_title is not None:
        cbar.set_label(cmap_title, fontsize = 16)


    if title is not None:
        plt.suptitle(title,fontsize=26)
    plt.show()
def plot_daley_globe(daley_lat,daley_lon, lon_swath, lat_swath,
                     nadir_data=None, lon_nadir=None, lat_nadir=None,
                     cmap = None, norm = None, levels = 0, variable = None,
                     title = None, unit = 'm', marker = None, log = False, 
                     figsize = (30, 10)):
    
    Nalong, Nacross = daley_lat.shape
    # swath_masked = mask_gap(swath_data)
    # lon_masked = mask_gap(lon_swath, fill_mean = True)
    # lat_masked = mask_gap(lat_swath, fill_mean = True)
    
    

    
    # Default settings if no cmap or norm are specified

    vmin = min([np.min(daley_lat[np.nonzero(daley_lat)]),np.min(daley_lon[np.nonzero(daley_lon)])])
    vmax = max([np.max(daley_lat),np.max(daley_lon)])

    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    # Initialize figure
    #fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.PlateCarree()})
    fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.Mollweide()})

    # Plot swath data
    
    c = axs[0].pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       daley_lat[:,:Nacross//2],  cmap = cmap,  vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs[0].pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       daley_lat[:,-Nacross//2:],  cmap = cmap,  vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    axs[0].set_title("Input Meridional length scales",fontsize=16)
    axs[0].coastlines()
    axs[0].set_global()
    
    c = axs[1].pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       daley_lon[:,:Nacross//2],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs[1].pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       daley_lon[:,-Nacross//2:],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    axs[1].set_title("Input Zonal length scales",fontsize=16)
    axs[1].coastlines()
    axs[1].set_global()
    
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)

    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])

    # Draw the colorbar
    cbar=fig.colorbar(c, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(f'length scale (m)', fontsize = 18)

    #cbar = fig.colorbar(c)

    if title is not None:
        plt.suptitle(title)
    plt.show()
    
    
    
def plot_norm_globe(norm_data, lon_swath, lat_swath,
                     nadir_data=None, lon_nadir=None, lat_nadir=None,
                     cmap = None, norm = None, levels = 0, variable = None,
                     title = None, unit = 'm', marker = None, log = False, 
                     figsize = (30, 10)):
    
    Nalong, Nacross = norm_data.shape
    # swath_masked = mask_gap(swath_data)
    # lon_masked = mask_gap(lon_swath, fill_mean = True)
    # lat_masked = mask_gap(lat_swath, fill_mean = True)
    
    

    
    # Default settings if no cmap or norm are specified

    vmin = -0.1#np.min(norm_data)
    vmax = 0.1#np.max(norm_data)

    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    # Initialize figure
    #fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.PlateCarree()})
    fig,axs = plt.subplots(figsize=figsize,ncols=1,subplot_kw={'projection': ccrs.Mollweide()})

    # Plot swath data
    
    c = axs.pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       norm_data[:,:Nacross//2],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs.pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       norm_data[:,-Nacross//2:],  cmap = cmap,vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    #axs.set_title("Meridional diffusivity",fontsize=16)
    axs.coastlines()
    axs.set_global()
    
    
    
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)

    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.3, 0.2, 0.4, 0.02])

    # Draw the colorbar
    cbar=fig.colorbar(c, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(f'Normalization factors (m\N{SUPERSCRIPT TWO})', fontsize = 16)

    #cbar = fig.colorbar(c)

    if title is not None:
        plt.suptitle(title,fontsize=18)
    plt.show()
    

def plot_error_region(error, lon_swath, lat_swath,
                     nadir_data=None, lon_nadir=None, lat_nadir=None,
                     cmap = None, norm = None, levels = 0, variable = None,
                     title = None,cmap_title=None, unit = 'm', marker = None, log = False, 
                     figsize = (30, 10),vmin=None,vmax=None,region="None",
                     lon_min=150,lon_max=220,lat_min=20,lat_max=70):
    
    Nalong, Nacross = error.shape
    # swath_masked = mask_gap(swath_data)
    # lon_masked = mask_gap(lon_swath, fill_mean = True)
    # lat_masked = mask_gap(lat_swath, fill_mean = True)
    
    # Default settings if no cmap or norm are specified
    if vmin is None:
        vmin = np.min(error)
    if vmax is None:
        vmax = np.max(error)
    #
    if region=="europe":
        lon_min=150
        lon_max=220
        lat_min=20
        lat_max=70
    elif region=="north_america":
        lon_min=140
        lon_max=5
        lat_min=8
        lat_max=70
    elif region=="south_america":    
        lon_min=150
        lon_max=90
        lat_min=-60
        lat_max=12
    elif region=="oceania" or region=="australia":    
        lon_min=270
        lon_max=354
        lat_min=-54
        lat_max=10
        
    elif region=="SEA" or region=="south_east_asia":    
        lon_min=250
        lon_max=324
        lat_min=-15
        lat_max=25
        
    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    # Initialize figure
    fig,axs = plt.subplots(figsize=figsize,ncols=1,subplot_kw={'projection': ccrs.PlateCarree()})
    #fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.Mollweide()})

    # Plot swath data
    
    c = axs.pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       error[:,:Nacross//2],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs.pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       error[:,-Nacross//2:],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    #axs.set_title("Train",fontsize=18)
    #axs.coastlines()
    #axs.set_global()
    axs.set_extent([lon_min,lon_max,lat_min,lat_max], ccrs.PlateCarree(central_longitude=180))
    
    
    
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.02)

    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.266, 0.2, 0.4673, 0.02])

    # Draw the colorbar
    cbar=fig.colorbar(c, cax=cbar_ax,orientation='horizontal')
    if cmap_title is not None:
        cbar.set_label(cmap_title, fontsize = 16)


    if title is not None:
        plt.suptitle(title,fontsize=26)
    plt.show()
def subplot_swath_samples(sample_list, nadir_sample_list = None, 
                          labels_list = [], variable = None, common_cmap = True,
                          cmap = None, norm = None, levels =0, title = None, unit = "m",
                          marker = None, log = False,
                          figsize = (30, 10)):
    swath_sample_list = copy.deepcopy(sample_list)
    # Scale for the unit
    if unit == 'cm':
        for sample in swath_sample_list:
            sample*=100
    if unit == 'mm':
        for sample in swath_sample_list:
            sample*=1000
    if variable is None:
        variable = "error (" + unit + ")" 
    
    
   # The total number of figures to be plotted sets the disposition
    fig_num = len(swath_sample_list) if common_cmap else len(swath_sample_list)
    fig, axs = plt.subplots(nrows = 1, ncols = fig_num, figsize = figsize)
    axs = axs.flatten() if fig_num > 1 else np.array([axs])

    
    # If no labels are given or not enough, empty labels are usd. 
    for i in range(len(swath_sample_list) - len(labels_list)):
        labels_list.append( ' ')   
    
    # If the plots share the same colormap, we neew to find the extrema of the 
    # list of grids
    if common_cmap:
        gridded_values = np.ma.array([swath_sample for swath_sample in swath_sample_list])
        vmin, vmax = np.min(gridded_values), np.max(gridded_values)
        # Default settings if no cmap or norm are specified
        cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels,log)
    else:
        cmap_init = cmap
        norm_init = norm
        
    # compute the distances
    along_track_dist, across_track_dist = distances(*swath_sample_list[0].shape, corners = True)
    
    
    # Each grid is plotted   
    for i  in range(len(swath_sample_list)):
        swath_sample = swath_sample_list[i]
        ax = axs[i]
        
        masked_sample = mask_gap(swath_sample)
        
        
        if not common_cmap:
            # Default settings if no cmap or norm are specified
            cmap, norm = get_cmap(np.min(swath_sample), np.max(swath_sample),
                                  cmap_init, norm_init, levels, log)
            
        c = ax.pcolormesh(across_track_dist, along_track_dist, masked_sample, 
                        cmap = cmap, norm=norm)
        
        if marker is not None:
            ax.scatter(marker[1], marker[0],marker='+',s=45,c='k')#, facecolors='none',
                      #  edgecolors='k', linewidth=3)
        
        if not common_cmap: 
            cbar = fig.colorbar(c, orientation='vertical', ax = ax, format="%2.f")
            cbar.set_label(variable,  fontsize=13)
        
        ax.set_title(labels_list[i],  fontsize=13)
        ax.set_ylabel("along track distance(km)",  fontsize=12)
        ax.set_xlabel("across track distance(km)",  fontsize=12)

    
    # FINAL LAYOUT ------------------------------------------------------------

    if common_cmap:
        # cbar_ax = axs[-1]
        cbar = fig.colorbar(c, orientation='vertical')
        cbar.ax.tick_params(labelsize = 12)
        cbar.set_label(variable,  fontsize=14)
   
    fig.set_tight_layout(True)      
    if title != None:
        fig.suptitle(title, fontsize=13)
    
    fig.show()
    #--------------------------------------------------------------------------


def distances(Nalong, Nacross, corners = False):
    res = 100/Nacross 
    if corners:
        pos_dist = np.linspace(10,60, int(Nacross/2)+1, endpoint = True)
        aacross_track_dist = np.concatenate((-np.flip(pos_dist), pos_dist))
        along_track_dist = np.arange(0, Nalong+1)*res
    else:
        pos_dist = np.arange(10+res/2, 60, res)
        aacross_track_dist = np.concatenate((-np.flip(pos_dist), np.zeros(1), pos_dist))
        along_track_dist = (np.arange(0, Nalong)+0.5)*res
    
    return along_track_dist, aacross_track_dist

def mask_gap(sample, fill_mean = False):
    Nalong, Nacross = sample.shape
    
    mask = np.concatenate(( np.zeros((Nalong, Nacross//2)), np.ones((Nalong, 1)),
                           np.zeros((Nalong, Nacross//2))), axis=1  )
    masked_sample = np.ma.masked_array(np.zeros((Nalong, Nacross+1)), mask=mask)
    masked_sample[:,:Nacross//2] = sample[:,:Nacross//2]
    masked_sample[:,-Nacross//2:] = sample[:,-Nacross//2:]
    if fill_mean:
        masked_sample[:, Nacross//2] = np.mean(sample[:,Nacross//2:Nacross//2+2],axis=1)
    
    return masked_sample

def segmented_norm( vmin, vmax, levels = 10, log = False):
    if vmin*vmax>=0:
        if log:
            bounds = np.logspace(np.log10(vmin), np.log10(vmax), levels)
        else:
            bounds = np.linspace(vmin, vmax, levels)
        return col.BoundaryNorm(boundaries=bounds, ncolors=256)
    else:
        vabs = np.max(np.abs([vmin,vmax]))
        bounds = np.linspace(-vabs, vabs, levels)
        return col.BoundaryNorm(boundaries=bounds, ncolors=256)
    
def get_cmap(vmin, vmax, cmap, norm, levels=0, log = False):
    # Default settings if no cmp or norm are specified
    if cmap is None:
        cmap = plt.cm.magma if vmin*vmax>0\
           else plt.cm.RdYlBu_r
    if norm is None:
        if levels == 0:
            vabs = np.max(np.abs([vmin,vmax]))
            norm = col.Normalize() if vmin*vmax>0\
                else col.Normalize(vmin=-vabs, vmax=vabs)
        else:
            norm = segmented_norm(vmin, vmax, levels = levels, log = log)
    return cmap, norm
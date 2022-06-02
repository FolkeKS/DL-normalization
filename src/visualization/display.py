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

    


def plot_variable_globes(var1,var2, lon_swath, lat_swath,
                     nadir_data=None, lon_nadir=None, lat_nadir=None,
                     cmap = None, norm = None, levels = 0, variable = None,
                     title = None,cmap_title=None, sub_title1=None,sub_title2=None,unit = 'm', marker = None, log = False, 
                     figsize = (30, 10),vmin=None,vmax=None,projection="PlateCarree"):
    
    Nalong, Nacross = var1.shape
    # swath_masked = mask_gap(swath_data)
    # lon_masked = mask_gap(lon_swath, fill_mean = True)
    # lat_masked = mask_gap(lat_swath, fill_mean = True)
    
    

    
    # Default settings if no cmap or norm are specified
    if vmin is None:
        vmin = min([np.min(var1),np.min(var2)])
    if vmax is None:
        vmax = max([np.max(var1),np.max(var2)])
    #

    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    # Initialize figure
    if projection=="PlateCarree":
        fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.PlateCarree()})
    elif projection=="Mollweide":
        fig,axs = plt.subplots(figsize=figsize,ncols=2,subplot_kw={'projection': ccrs.Mollweide()})
    else:
        raise NotImplementedError("Available: PlateCarree, Mollweide")

    # Plot swath data
    
    c = axs[0].pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       var1[:,:Nacross//2],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs[0].pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       var1[:,-Nacross//2:],  cmap = cmap, vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    axs[0].set_title(sub_title1,fontsize=18)
    axs[0].coastlines()
    axs[0].set_global()
    
    c = axs[1].pcolormesh(lon_swath[:,:Nacross//2], lat_swath[:,:Nacross//2],
                       var2[:,:Nacross//2],  cmap = cmap,  vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    
    c = axs[1].pcolormesh(lon_swath[:,-Nacross//2:], lat_swath[:,-Nacross//2:],
                       var2[:,-Nacross//2:],  cmap = cmap,  vmin=vmin,vmax=vmax, 
                       transform=ccrs.PlateCarree(central_longitude=180))
    axs[1].set_title(sub_title2,fontsize=18)
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
                     figsize = (30, 10),vmin=None,vmax=None,projection="PlateCarree"):
    
    Nalong, Nacross = error.shape

    # Default settings if no cmap or norm are specified
    if vmin is None:
        vmin = np.min(error)
    if vmax is None:
        vmax = np.max(error)
    #

    cmap, norm = get_cmap(vmin, vmax, cmap, norm, levels, log)
    # Initialize figure
    if projection=="PlateCarree":
        fig,axs = plt.subplots(figsize=figsize,ncols=1,subplot_kw={'projection': ccrs.PlateCarree()})
    elif projection=="Mollweide":
        fig,axs = plt.subplots(figsize=figsize,ncols=1,subplot_kw={'projection': ccrs.Mollweide()})
    else:
        raise NotImplementedError("Available: PlateCarree, Mollweide")

    # Plot swath data
    #print(axs)
    
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
    
    
def plot_norm_globe(norm_data, lon_swath, lat_swath,
                     nadir_data=None, lon_nadir=None, lat_nadir=None,
                     cmap = None, norm = None, levels = 0, variable = None,
                     title = None, unit = 'm', marker = None, log = False, 
                     figsize = (30, 10)):
    
    Nalong, Nacross = norm_data.shape

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
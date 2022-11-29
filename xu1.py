
# checked file: gefs_90_DL_training.py
# from xu import ndates,Ds,mmean,rmse,bias,cf,atxt
# from xu import wjson
# from xu import rgz

# xu function set: {'mmean', 'atxt', 'rmse', 'cf', 'wjson', 'bias', 'ndates', 'rgz', 'Ds'}

import numpy as np
import matplotlib.pyplot as plt
def mmean(D, meanaxis=0, keepdims=False):
    """meanaxis   {int, tuple of int, None}, axis=-1 the last dim """
    return np.nanmean(D, axis=meanaxis, keepdims=keepdims)

def atxt(x=0, y=-.15, txt='xxx', **kwargs):
    '''plot txt at axes (x,y) axes coodinate'''
    plt.gca().text(x, y, txt, **kwargs)

def rmse(D, Dtrue, axis=0):
    '''return RMSE along axis(default 0, time, could be tuple)'''
    A = D - Dtrue
    return np.sqrt(mmean(A * A, meanaxis=axis))

def cf(
        lon,
        lat,
        M,
        lev=25,
        cm='bwr',
        title='',
        l='',
        r='',
        cbar=True,
        cax=[0, -0.05, 1, 0.03],  #[left, bottom, width, height]
        bottom=0,
        clabel='',
        cticks=False,
        figsize=(6, 4.5),
        extent=None,
        aspect=1.3,
        cyclic=False,
        omask=False,
        lmask=False,
        **kwargs):
    '''kwargs:
    norm=n,
    alpha=0.8,
    extend='both','max','min' 'neither'
    extent=[-125, -65, 25, 45.7]
    cax=[1.02, 0.0, 0.03, 1] for horizontal cbar
             '''
    import cartopy.crs as ccrs
    from xu import lat_lon_ticklabels
    #plt.switch_backend('agg')

    if cyclic:
        from cartopy.util import add_cyclic_point
        M, lon = add_cyclic_point(M, coord=lon)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(), aspect=aspect)
    CS = ax.contourf(lon, lat, M, levels=lev, cmap=cm, **kwargs)

    import cartopy.feature as cfeature
    if omask:
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=1)
    if lmask:
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)

    # ax.coastlines(resolution='50m')
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=1, alpha=0.8)
    ax.add_feature(cfeature.STATES, linestyle=':', lw=0.5)

    if title: ax.set_title(title)
    if l: ax.set_title(l, loc='left')
    if r: ax.set_title(r, loc='right')
    if bottom: fig.subplots_adjust(bottom=bottom)

    if cbar:
        #cb_ax = fig.add_axes([0.15, 0.16, 0.73,0.03])  #[left, bottom, width, height]
        cb_ax = ax.inset_axes(cax, transform=ax.transAxes)
        cbar = plt.colorbar(CS,
                            cax=cb_ax,
                            orientation='horizontal',
                            extendrect=False,
                            extendfrac='auto',
                            label=clabel)

    if cticks:
        cbar.set_ticks(lev)
        cbar.set_ticklabels(lev)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    return ax,CS

def wjson(obj, fn='data.json'):
    '''dump to json file'''
    import json
    try:
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=4)
            print('write ' + fn + ' scuccssful!')
    except IOError as e:
        print(e)

def bias(Df,Do,axis=0):
    return mmean(Df,axis)-mmean(Do,axis)

def ndates(nhours=0, newdate=None, fmt='%Y%m%d') -> str:
    ''' improved ndate()  fmt='%Y%m%d%H', defulat 00z
    return str,  accord to fmt='%Y %m %d %H %h'
    %b  Sep
    %j  251 Day of year as zero-padded
    %W  weeek of year (Monday as the first day of week'''
    from datetime import datetime
    from datetime import timedelta
    if newdate is None:
        today = datetime.now()
    else:
        try:
            today = datetime.strptime(newdate, '%Y%m%d')
        except:
            print('error date:', newdate)
            today = datetime.strptime(newdate[:6] + '28', '%Y%m%d')

    delta = today + timedelta(hours=nhours)
    return delta.strftime(fmt)

def rgz(file='filename.gz', shape=(480, 1440)):
    '''only work for 1 file gz'''
    import gzip
    with gzip.open(file, 'rb') as f:
        D = np.frombuffer(f.read(), dtype='<f4').reshape(shape)
    return D

def Ds(ds_file="data/nldas_smp.nc"):
    """return dataset from nc_file, by xarray
    need to be "good" nc"""
    import xarray as xr

    if ds_file.endswith(("grb", "grb2")):
        Ds = xr.open_dataset(ds_file, engine="cfgrib")
    else:
        Ds = xr.open_dataset(ds_file)
    Ds  # display information
    return Ds


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 00:57:42 2022
@author: li.xu

contact:
    Li Xu
    CPC/NCEP/NOAA
    (301)683-1548
    li.xu@noaa.gov

"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import xu
from xu import ndates,Ds,mmean,rmse,bias,cf,atxt


#EFSv12 reforecast:

f1='/cpc/gth/GTH_DATABASE/WEEK2_HEAT/GEFS_SEHOS/GEFS_v12_reforecast_HeatIndex.nc'
f2='/cpc/gth/GTH_DATABASE/WEEK2_HEAT/GEFS_SEHOS/GEFS_v12_reforecast_airT.nc'



#CDAS reanalysis:

o2='/cpc/home/evan.oswald/R1_SEHOS/CDAS_1x1_reanalysis_airT.nc'
o1='/cpc/home/evan.oswald/R1_SEHOS/CDAS_1x1_reanalysis_HeatIndex.nc'



dso1=Ds(o1)
dsf1=Ds(f1)


# %%
#dsf1=dsf1.set_coords('issue_date')
#dso1=dso1.set_coords('issue_date')


# import cartopy.crs as ccrs

# p=dsf1.max_heat_index_2m.isel(time=0,ensemble=0,fcst_date=0).T.plot(subplot_kws=dict(projection=ccrs.PlateCarree()))
# p.axes.coastlines()



lat=dsf1.latitude.values
lon=dsf1.longitude.values+0.5 # slight shift right half degree

to=dso1.issue_date.values.astype('i4')
tf=dsf1.issue_date.values.astype('i4')




f=dsf1.max_heat_index_2m.values
f=mmean(f,2)  #[71,31,7,5280]

o=dso1.max_heat_index_2m.values
o=np.flip(o,axis=1) # to make latitude from low to high, the same as fcst


def p_rmse(M,title):
    lev=np.arange(1,15)
    cf(lon,lat,M.T,lev,cm='YlGnBu',extend='both',title=title,l='HI>80',r='unit:F')
    m=mmean(M,(0,1))
    atxt(-75,30,f'mean={m:.2f}')

for i in range(7): #day8-14  start from 0


    tfn=tf+7+i #day

    select=np.in1d(to,tfn) # to get same issue_date at observation with forecast
    oday=o[:,:,select]

    bias_day=bias(f[:,:,i,:],oday,axis=2)
    rmse_day=rmse(f[:,:,i,:],oday,axis=2)

    fc=f[:,:,i,:]-bias_day[:,:,np.newaxis]

    rmse_fc=rmse(fc,oday,axis=2)


    lev=[-5,-4,-3,-2,-1,1,2,3,4,5]

    cf(lon,lat,bias_day.T,lev,extend='both',title=f'bias for day{i+7+1} \n',l='heat index',r='unit:F')
    m=mmean(bias_day,(0,1))
    atxt(-75,30,f'mean={m:.2f}')

    p_rmse(rmse_day,f'RMSE for day{i+7+1} \n')
    p_rmse(rmse_fc,f'RMSE after BC for day{i+7+1} \n')



# for HI>80
tfn=tf+7 #day8

select=np.in1d(to,tfn) # to get same issue_date at observation with forecast
o8=o[:,:,select]


o8_80=o8>80
o8[~o8_80]=np.nan
f8=f[:,:,0,:]
f8[~o8_80]=np.nan



bias8=bias(f8,o8,axis=2)
rmse8=rmse(f8,o8,axis=2)

fc=f8-bias8[:,:,np.newaxis]

rmse8_fc=rmse(fc,o8,axis=2)


lev=[-5,-4,-3,-2,-1,1,2,3,4,5]

cf(lon,lat,bias8.T,lev,extend='both',title='bias for day8  \n',l='HI>80',r='unit:F')
m=mmean(bias8,(0,1))
atxt(-75,30,f'mean={m:.2f}')



lev=np.arange(15)
cf(lon,lat,rmse8.T,lev,cm='YlGnBu',extend='both',title='RMSE for day8 \n',l='HI>80',r='unit:F')
cf(lon,lat,rmse8_fc.T,lev,cm='YlGnBu',extend='both',title='RMSE for day8 \n',l='HI>80',r='unit:F')





p_rmse(rmse8,'RMSE for day8 \n')
p_rmse(rmse8_fc,'RMSE after BC for day8 \n')




def M2V(D):
    '''convert from 2D Matrix to 1D vector, eliminate all undefined ocean
    [...,31,71] to [...,1368]'''
    mask=np.load('mask.npy')
    return D[...,mask]



def V2M(V):
    '''convert from 1D vector to 2D matrix for map, fill ocean with undefined'''

    def npa(shape=(60, 150), default=np.nan):
        """create a float32 ndarray with nan inital"""
        # return np.zeros(shape, dtype='float32') * np.nan
        return np.full(shape, default, dtype="float32")

    mask=np.load('mask.npy')
    if V.shape[-1] == 1368:
        M = npa((*V.shape[:-1], 31, 71))
        M[..., mask] = V
        return M
    else:
        print('wrong shape (need to be 1368 at last dim)')
        return None


def test():
    x=M2V(D.T)
    N=V2M(x)




def get_mask():
    D=dsf1['max_heat_index_2m'].values
    M=D[:,:,0,0,0].T
    mask=~np.isnan(M)  #total 1368 land points
    np.save('mask.npy',mask)


def num2date(n) -> str :
    from datetime import datetime
    from datetime import timedelta
    start=datetime.strptime('00010101','%Y%m%d')
    date=start+timedelta(days=n-367)  # to 0000 year jan 0
    return date.strftime('%d%h%Y')


def date2num(yyyymmdd='20210101') -> int:
    from datetime import datetime
    from datetime import timedelta
    end=datetime.strptime(yyyymmdd,'%Y%m%d')
    start=datetime.strptime('00010101','%Y%m%d')
    return (end-start).days+367   # from 0000 jan 0






def main(yyyymmdd):
    print(yyyymmdd)



def read_hourp(yyyymmdd):
    ''' read 0.25 hourly p analysis from Shaorong Wu

    DSET  ^../%y4/%y4%m2/CONUS_GAUGE_PRCP_HLY_0.25deg_NG8000_ADJ.lnx.%y4%m2%d2
*
options   template
*
UNDEF  -999.0
*
TITLE  CONUS Hourly Gauge Precipitation Analysis adjusted against CPC daily
*
XDEF  260 LINEAR -129.875 0.25
*
YDEF  128 LINEAR  20.125  0.25
*
ZDEF 1 LEVELS  1
*
TDEF 999999 LINEAR 00Z1JAN1948 1hr
*
VARS 2
prcp   1 00 precipitation in mm/hr
samp   1 00 number of reporting gauges
ENDVARS
'''


    p=f'/cpc/sate/XIE/BASE/DSI3240/ANA/{yyyymmdd[:4]}/{yyyymmdd[:6]}/CONUS_GAUGE_PRCP_HLY_0.25deg_NG8000_ADJ.lnx.{yyyymmdd}.gz'

    from xu import rgz
    D=rgz(p,(24,2,128,260))
    Da=np.copy(D[:,0])
    Da[Da<-900]=np.nan

    return Da  #unit: mm/hr




if __name__ == '__main__':
    import sys
    import fire
    if len(sys.argv) <= 1:
        print('input: YYYYMMDD or -h')
    else:
        if len(sys.argv[1]) == 8:
            main(sys.argv[1])
        else:
            fire.Fire()

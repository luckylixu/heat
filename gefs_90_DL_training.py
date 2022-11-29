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


from xu import ndates,Ds,mmean,rmse,bias,cf,atxt


#EFSv12 reforecast:

f1='/cpc/gth/GTH_DATABASE/WEEK2_HEAT/GEFS_SEHOS/GEFS_v12_reforecast_HeatIndex.nc'
f2='/cpc/gth/GTH_DATABASE/WEEK2_HEAT/GEFS_SEHOS/GEFS_v12_reforecast_airT.nc'



#CDAS reanalysis:

o2='/cpc/home/evan.oswald/R1_SEHOS/CDAS_1x1_reanalysis_airT.nc'
o1='/cpc/home/evan.oswald/R1_SEHOS/CDAS_1x1_reanalysis_HeatIndex.nc'



dso1=Ds(o1)
dsf1=Ds(f1)

dso2=Ds(o2)
dsf2=Ds(f2)
# %%
lat=dsf1.latitude.values
lon=dsf1.longitude.values+0.5 # slight shift right half degree

to=dso1.issue_date.values.astype('i4')
tf=dsf1.issue_date.values.astype('i4')




f=dsf1.max_heat_index_2m.values
fhi=mmean(f,2)  #[71,31,7,5280] ensemble mean

o=dso1.max_heat_index_2m.values
ohi=np.flip(o,axis=1) # to make latitude from low to high, the same as fcst



f=dsf2.max_air_temp_2m.values
ftx=mmean(f,2)  #[71,31,7,5280] ensemble mean

o=dso2.max_air_temp_2m.values
otx=np.flip(o,axis=1) # to make latitude from low to high, the same as fcst


# %%

i=1

tfn=tf+7+i #fcst day

select_o=np.in1d(to,tfn) # to get same issue_date at observation with forecast
after2000=tfn>730485





mask90=np.logical_and(ftx[:,:,i]>90,~np.isnan(otx[:,:,select_o]))
mask90=np.logical_and(mask90,after2000)



fhi90=fhi[:,:,i,:][mask90]
ftx90=ftx[:,:,i,:][mask90]

otx90=otx[:,:,select_o][mask90]
ohi90=ohi[:,:,select_o][mask90]


bias=ftx90-otx90
mmean(bias)
mmean(np.abs(bias))


test=np.ones_like(fhi[:,:,i])*np.nan
test[mask90]=bias
bias_conus=mmean(test,2)
abias_conus=mmean(np.abs(test),2)

def p_(M,title):
    lev=np.linspace(-10,10,21)
    cf(lon,lat,M.T,lev,extend='both',title=title,l='Tmax>90',r='unit:F')
    m=mmean(M,(0,1))
    atxt(-75,30,f'mean={m:.2f}')

p_(bias_conus,'Bias for day8 \n')
p_(abias_conus,'Abs_error for day8 \n')




def dem(var='tx'):
    '''var: tx,hi'''
    D=[]
    for nday in range(8,15):
        f=f'DEM/DEM_{var}_day{nday}.npy'
        D.append(np.load(f).T)

    D=np.asarray(D)
    return D


def oni(yyyymmdd='20210501'):
    yyyy=yyyymmdd[:4]
    mm=yyyymmdd[4:6]
    url='https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php'

    import pandas as pd
    tables=pd.read_html(url)
    table=tables[8]

    # remove year rows
    #indx=list(range(0,80,11))
    #x=table.drop(table.index[indx])
    x=table[table[0].str.contains('Year') == False]
    oni=x[int(mm)][x[0]==yyyy].values[0]

    return np.asarray(float(oni))





# %% input
year= (tfn/365.25).astype('i4')
doy=(tfn-year*365.25).astype('i4')
daug1=(doy-214)/365     #[5280]



daug=np.ones_like(fhi)*daug1
dlat=np.ones_like(fhi)*lat[np.newaxis,:,np.newaxis,np.newaxis]
dlon=np.ones_like(fhi)*lon[:,np.newaxis,np.newaxis,np.newaxis]


aug90=daug[:,:,i][mask90]
lat90=dlat[:,:,i][mask90]
lon90=dlon[:,:,i][mask90]


data=np.load('ssmi_oni.npy.npz') #2000-2019
ssmi=data['ssmi'].T
oni=data['oni']
issue_date=data['issue_date']

select1=np.in1d(tfn,issue_date)
select2=np.in1d(issue_date,tfn)

ssmi1=np.ones_like(fhi[:,:,i])*np.nan
ssmi1[:,:,select1]=ssmi[:,:,select2]

oni1=np.ones_like(fhi[:,:,i])*np.nan
oni1[:,:,select1]=oni[select2]



# %%



# for tx
y=otx[:,:,select_o][mask90]


Ddem=dem('tx')
dem1=np.ones_like(fhi[:,:,i]).T*Ddem[i]
x1=ftx[:,:,i][mask90]
x2=fhi[:,:,i][mask90]
x3=dlat[:,:,i][mask90]
x4=daug[:,:,i][mask90]
x5=oni1[mask90]
x6=ssmi1[mask90].astype('f8')
x7=dem1.T[mask90]

x=np.asarray([x1,x2,x3,x4,x5,x6,x7]).T

x[np.isnan(x)]=0

# %%
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

model = keras.models.load_model('model/GEFSV12_tx90.h5')
model.summary()



history = model.fit(
    x,
    y,
    epochs=5000,
    validation_split=0.2,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)],
)


from xu import wjson

wjson(history.history, f"test1.json")
model.save('model/GEFSV12_tx90.h5')


plt.plot(history.history["mae"], label="MAE (training data)")
plt.plot(history.history["val_mae"], label="MAE (validation data)")
plt.title("MAE for deep learning Tmax >90")
plt.ylabel("MAE value")
plt.xlabel("No. epoch")
plt.legend(loc="best")

plt.show()


# hi
Ddem=dem('hi')
dem1=np.ones_like(fhi[:,:,i]).T*Ddem[i]
x1=ftx[:,:,i][mask90]
x2=fhi[:,:,i][mask90]
x3=dlat[:,:,i][mask90]
x4=daug[:,:,i][mask90]
x5=oni1[mask90]
x6=ssmi1[mask90].astype('f8')
x7=dem1.T[mask90]

x=np.asarray([x1,x2,x3,x4,x5,x6,x7]).T

x[np.isnan(x)]=0

y=ohi[:,:,select_o][mask90]


model = keras.models.load_model('model/GEFSV12_hi90.h5')
model.summary()



history = model.fit(
    x,
    y,
    epochs=5000,
    validation_split=0.2,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)],
)

model.save('model/GEFSV12_hi90.h5')




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


def test():

    from tensorflow import keras

    print("keras version:", keras.__version__)

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(7,)),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dense(5, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(3, activation="relu"),
            keras.layers.Dense(2, activation="relu"),
            keras.layers.Dense(1)
        ]
    )

    optimizer = keras.optimizers.RMSprop(0.0001)

    model.compile(loss="mae", optimizer=optimizer, metrics=["mae", "mse"])

    model.summary()

    model.save('model/GEFSV12_tx90.h5')


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

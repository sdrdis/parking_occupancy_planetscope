import urllib.request
from datetime import timedelta, date
from os.path import join, isdir, isfile
import os
import time
import random

# https://oceancitylive.com/ocean-city-webcams/ocean-city-inlet-parking-lot-cam/
# https://www.google.com/maps/place/Hugh+T.+Cropper+Inlet+Parking+Lot/@38.325422,-75.086982,16z/data=!4m5!3m4!1s0x0:0xfd86dbe123aa7d92!8m2!3d38.3254224!4d-75.086982?hl=fr-FR


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = date(2019, 3, 9)
end_date = date(2020, 2, 7)
for single_date in daterange(start_date, end_date):
    date_str = single_date.strftime("%Y-%m-%d")
    print (date_str)
    dat_filepath = 'data/' + date_str + '.dat'
    mp4_filepath = 'data/' + date_str + '.mp4'
    if (isfile(dat_filepath)):
        continue
    try:
        urllib.request.urlretrieve('https://s33.ipcamlive.com/timelapses/5aec8d9905b9a/'+date_str+'/video.mp4', mp4_filepath)
        urllib.request.urlretrieve('https://s33.ipcamlive.com/timelapses/5aec8d9905b9a/'+date_str+'/video.dat', dat_filepath)
    except:
        continue
        
    time.sleep(random.randrange(3, 5))
'''

from_date = '2019-03-09'




'''
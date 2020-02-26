import json
import numpy as np
from datetime import datetime
import pytz
import numpy as np
import os
from os.path import join

# METHOD 1: Hardcode zones:
zone = pytz.timezone('UTC')

#path = 'planet_data/extracted_data'
path = 's1_data/site_1_stabilized_f'

list_dates = []
for item in os.listdir(path):
    datetime_str = item.split('_')[0] + '_' + item.split('_')[1]
    datetime_str = datetime_str.split('.')[0]
    print (datetime_str)
    dt = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
    dateutc = zone.localize(dt)
    print (dateutc)
    list_dates.append([dateutc, item])

np.save('dates.npy', list_dates)
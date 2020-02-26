import json
import numpy as np
from datetime import datetime
import pytz
import numpy as np

# METHOD 1: Hardcode zones:
zone = pytz.timezone('UTC')

# utc = datetime.utcnow()


with open('sentinel2_data/response.json', 'r') as f:
    data = json.load(f)['tiles']

list_dates = []
for item in data:
    t = item['sensingTime']
    
    utc = datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
    dateutc = zone.localize(utc)
    list_dates.append(dateutc)

np.save('dates.npy', list_dates)
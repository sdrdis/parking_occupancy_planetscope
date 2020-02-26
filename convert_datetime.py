from datetime import datetime
import pytz

# METHOD 1: Hardcode zones:
from_zone = pytz.timezone('UTC')
to_zone = pytz.timezone('US/Eastern')

# utc = datetime.utcnow()
utc = datetime.strptime('2019-07-10T16:02:43Z', '%Y-%m-%dT%H:%M:%SZ')

dateutc = from_zone.localize(utc)
dateeastern = dateutc.astimezone(to_zone)

'''
# Tell the datetime object that it's in UTC time zone since 
# datetime objects are 'naive' by default
utc = utc.replace(tzinfo=from_zone)

# Convert time zone
central = utc.astimezone(to_zone)
'''
print (dateutc)
print (dateeastern)
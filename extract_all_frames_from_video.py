import cv2
import json
import numpy as np
import pytz
from datetime import datetime
from os.path import join, isfile

def get_frames(dat_filepath, to_zone):
    with open(dat_filepath, 'r') as f:
        data = f.read().split('\n')
        
    if (data[0] != 'version=1' or data[1] != 'fps=8' or data[2] != 'begin'):
        raise Exception('Unknown format!')
        
    min_diff = None
    dates = []
    for i in range(0, len(data) - 4):
        frame_date = to_zone.localize(datetime.strptime(data[i + 3], '%Y-%m-%d %H:%M:%S'))
        dates.append(frame_date)
            
    return dates

to_zone = pytz.timezone('US/Eastern')
videos_path = 'data'

date_str = '2019-07-20'

    
dat_filepath = join(videos_path, date_str+'.dat')
fs = get_frames(dat_filepath, to_zone)
    
        
# Opens the Video file
cap= cv2.VideoCapture('data/'+date_str+'.mp4')

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    fileid = fs[i].strftime('%Y%m%d_%H%M%S')
    print (fileid)
    cv2.imwrite('tmp/'+fileid+'.png',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()
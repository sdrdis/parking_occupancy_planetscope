import cv2
import json
import numpy as np
import pytz
from datetime import datetime
from os.path import join, isfile

def get_framenumber(dat_filepath, date, to_zone):
    with open(dat_filepath, 'r') as f:
        data = f.read().split('\n')
        
    if (data[0] != 'version=1' or data[1] != 'fps=8' or data[2] != 'begin'):
        raise Exception('Unknown format!')
        
    min_diff = None
    min_i = 0
    for i in range(0, len(data) - 4):
        frame_date = to_zone.localize(datetime.strptime(data[i + 3], '%Y-%m-%d %H:%M:%S'))
        diff = abs(frame_date - date)
        if (min_diff is None or min_diff > diff):
            min_diff = diff
            min_i = i
            
    return min_i

to_zone = pytz.timezone('US/Eastern')
videos_path = 'data'

dates = np.load('dates.npy')

for infos in dates:
    _date = infos[0]
    filename = infos[1]
    to_filepath = 'extracts_s1/'+filename[:-4] +'.png'
    if (isfile(to_filepath)):
        continue
    date = _date.astimezone(to_zone)
    date_video = date.strftime('%Y-%m-%d')
    time_video = date.strftime('%Y-%m-%d %H:%M:%S')
    print (_date, date_video, time_video)
    
    dat_filepath = join(videos_path, date_video + '.dat')
    
    if (not isfile(dat_filepath)):
        print ('NO DAT FILE!')
        continue
    
    target_frame = get_framenumber(dat_filepath, date, to_zone)
        
    # Opens the Video file
    cap= cv2.VideoCapture('data/'+date_video+'.mp4')
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if (target_frame == i):
            cv2.imwrite('extracts_s1/'+filename[:-4] +'.png',frame)
        i+=1
     
    cap.release()
    cv2.destroyAllWindows()
import pandas as pd

name = 'Avon Street CP'
filepath = 'BANES_Historic_Car_Park_Occupancy.csv'

df = pd.read_csv(filepath)
sel_df = df[df['Name'] == name]

sel_df['LastUpdate'] = pd.to_datetime(sel_df['LastUpdate'])
sel_df.sort_values(by=['LastUpdate'])
sel_df.to_pickle('data.pkl')
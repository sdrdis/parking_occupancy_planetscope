import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_pickle('data.pkl')
df = df.sort_values(by=['LastUpdate'])
print (df)
print (df['LastUpdate'])
print (df['Occupancy'])
plt.plot(df['LastUpdate'].tolist(), (df['Occupancy'] / df['Capacity']).tolist())

plt.show()
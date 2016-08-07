
import matplotlib.pyplot as plt
import numpy as np
from ann import BPNet
import pandas as pd

def importdata():
    dateparse = lambda dates: pd.datetime.strptime(str(dates), '%Y%m%d')
    data = pd.read_csv('../data/count_table.csv', parse_dates='ds',index_col='ds', date_parser=dateparse)
    artists = [i[0] for i in data.groupby('artist_id')]
    return data,artists


data,artists = importdata()
one_data = data[data.artist_id == artists[33]]
max = one_data["count(1)"].max()
min = one_data["count(1)"].min()

# 100

Y = [(i-min)/(max-min) if i else 0 for i in one_data["count(1)"]]
# print(Y)

# Y = [(np.math.sin(float(i/10))+1)/2 for i in range(1000)]

X=[]
y = []
for i in range(20,len(Y)-1,1):
    X.append(Y[i-20:i])
    y.append(Y[i:i+1])
bp = BPNet.BPNet([20, 15,2, 1], True)


for i in range(1000):
    bp.training(np.array(X[:-60]),np.array(y[:-60]))
out = [i[0] for i in bp.run(X[:-60])]

for i in range(len(X[-60:])):
    out.append(bp.run(np.array([out[-20:],]))[0][0])

plt.plot(y)
plt.plot(out)
plt.show()






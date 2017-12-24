import pandas as pd

theta_0 = pd.read_csv('theta_0.csv', sep='=')['value'].values
mkt_data = pd.read_csv('mkt_data.csv').values
data = pd.read_csv('additional_data.csv', sep='=')['value'].values
S_0 = data[0]
r = data[1]
M = data[2]
deg = int(data[3])
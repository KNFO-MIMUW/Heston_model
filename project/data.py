import numpy as np
import heston_functions as hf
import pandas as pd

theta_0 = np.array([0.07, 0.11, -0.75, 3.15, 0.15])
theta = np.array([0.08, 0.10, -0.8, 3, 0.25])
data = np.array([100, 0.02, 100, 64]) # S_0, r, M, deg
S_0 = 100
r = 0.02
M = 100
deg = 64
mkt_data = np.vstack(([hf.eur_call_heston_price(theta, 80, 0.25, data[0], data[1], data[2], 64), 80, 0.25],
                      [hf.eur_call_heston_price(theta, 80, 0.5, data[0], data[1], data[2], 64), 80, 0.5],
                      [hf.eur_call_heston_price(theta, 80, 1, data[0], data[1], data[2], 64), 80, 1],
                      [hf.eur_call_heston_price(theta, 80, 2, data[0], data[1], data[2], 64), 80, 2],
                      [hf.eur_call_heston_price(theta, 90, 0.25, data[0], data[1], data[2], 64), 90, 0.25],
                      [hf.eur_call_heston_price(theta, 90, 0.5, data[0], data[1], data[2], 64), 90, 0.5],
                      [hf.eur_call_heston_price(theta, 90, 1, data[0], data[1], data[2], 64), 90, 1],
                      [hf.eur_call_heston_price(theta, 90, 2, data[0], data[1], data[2], 64), 90, 2],
                      [hf.eur_call_heston_price(theta, 100, 0.25, data[0], data[1], data[2], 64), 100, 0.25],
                      [hf.eur_call_heston_price(theta, 100, 0.5, data[0], data[1], data[2], 64), 100, 0.5],
                      [hf.eur_call_heston_price(theta, 100, 1, data[0], data[1], data[2], 64), 100, 1],
                      [hf.eur_call_heston_price(theta, 100, 2, data[0], data[1], data[2], 64), 100, 2],
                      [hf.eur_call_heston_price(theta, 110, 0.25, data[0], data[1], data[2], 64), 110, 0.25],
                      [hf.eur_call_heston_price(theta, 110, 0.5, data[0], data[1], data[2], 64), 110, 0.5],
                      [hf.eur_call_heston_price(theta, 110, 1, data[0], data[1], data[2], 64), 110, 1],
                      [hf.eur_call_heston_price(theta, 110, 2, data[0], data[1], data[2], 64), 110, 2],
                      [hf.eur_call_heston_price(theta, 120, 0.25, data[0], data[1], data[2], 64), 120, 0.25],
                      [hf.eur_call_heston_price(theta, 120, 0.5, data[0], data[1], data[2], 64), 120, 0.5],
                      [hf.eur_call_heston_price(theta, 120, 1, data[0], data[1], data[2], 64), 120, 1],
                      [hf.eur_call_heston_price(theta, 120, 2, data[0], data[1], data[2], 64), 120, 2],
                      ))

mkt_data = pd.DataFrame(mkt_data)
mkt_data.columns = ['price', 'K', 'T']
pd.DataFrame(mkt_data).to_csv('mkt_data.csv', index=False, sep=',', decimal='.')
theta = pd.DataFrame({'parameter': ['v_0', 'bar_v', 'rho', 'kappa', 'sigma'], 'value': theta})
pd.DataFrame(theta).to_csv('theta.csv', index=False, sep='=', decimal='.')
theta_0 = pd.DataFrame({'parameter': ['v_0', 'bar_v', 'rho', 'kappa', 'sigma'], 'value': theta_0})
pd.DataFrame(theta).to_csv('theta_0.csv', index=False, sep='=', decimal='.')
data = pd.DataFrame({'parameter': ['S_0', 'r', 'M', 'deg'], 'value': data})
pd.DataFrame(data).to_csv('additional_data.csv', index=False, sep='=', decimal='.')

import numpy as np
import math
import heston_functions as hf

class option():
    def __init__(self, theta, data):
        self.theta = theta
        self.S_0 = data[0]
        self.r = data[1]
        self.M = int(data[2])
        self.deg = int(data[3])
        self.K = data[4]
        self.T = data[5]
        self.N = int(data[6])
        self.it = int(data[7])
    def path(self):
        dt = self.T/self.N
        paths = np.random.multivariate_normal([0, 0], [[1, self.theta[2]], [self.theta[2], 1]], self.N + 1).transpose()
        paths[0, 0] = self.S_0
        paths[1, 0] = self.theta[0]
        for j in range(0, self.N):
            paths[1, j + 1] = paths[1, j] + self.theta[3]*(self.theta[1] - max(paths[1, j], 0))*dt \
                               + self.theta[4]*math.sqrt(max(0, paths[1, j])*dt)*paths[1, j + 1]
            paths[0, j + 1] = paths[0, j]*math.exp((self.r - max(paths[1, j], 0)/2)*dt \
                                                 + math.sqrt(max(paths[1, j], 0)*dt)*paths[0, j + 1])
        return paths
    def step(self, price, vol):
        dt = self.T / self.N
        norm_var = np.random.multivariate_normal([0, 0], [[1, self.theta[2]], [self.theta[2], 1]], self.it).transpose()
        vol[vol < 0] = 0
        sqrt_vol = np.sqrt(vol)
        next_vol = vol + self.theta[3] * (self.theta[1] - vol) * dt \
                   + self.theta[4] * sqrt_vol * math.sqrt(dt) * norm_var[0, :]
        exp_vol = (self.r - vol / 2) * dt + sqrt_vol * math.sqrt(dt) * norm_var[1, :]
        next_price = price * np.exp(exp_vol)
        return np.concatenate([[next_vol], [next_price]])
    def eur_option(self):
        return hf.eur_call_heston_price(self.theta, self.K, self.T, self.S_0, self.r, self.M, self.deg)
    def asian_option(self):
        price = np.empty(self.it)
        price.fill(self.S_0)
        vol = np.empty(self.it)
        vol.fill(self.theta[0])
        sum = np.zeros(self.it)
        for j in range(0, self.N):
            next = self.step(price, vol)
            sum = sum + next[1, :]
            vol = next[0, :]
            price = next[1, :]
        asian_price = sum/self.N - self.K
        asian_price[asian_price < 0] = 0
        return asian_price.sum()*math.exp(- self.r*self.T)/self.it

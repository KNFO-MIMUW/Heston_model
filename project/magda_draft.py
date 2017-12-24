import numpy as np
import math

theta = np.array([0.08, 0.10, -0.8, 3, 0.25])

def path(theta, N, S_0, T, er): #
    dt = T/N

    # generate two vectors containing standard normal draws (correlated)
    st_norm_draws = np.random.multivariate_normal([0, 0], [[1, theta[2]], [theta[2], 1]], N+1)

    volatility_value = np.arange(N+1, dtype='double')
    asset_price = np.arange(N+1, dtype='double')

    volatility_value[0] = theta[0]
    asset_price[0] = S_0

    for j in range(0, N):
       # calculate volatility values from draws and previous vols
       volatility_value[j+1] = volatility_value[j] + theta[3]*(theta[1] - max(volatility_value[j], 0))*dt \
                               + theta[4]*math.sqrt(max(0,volatility_value[j]))*st_norm_draws[j, 0]*math.sqrt(dt)

       # calculate the next asset value from the volatility path and from draws
       asset_price[j+1] = asset_price[j]*math.exp((er - max(volatility_value[j],0))/2*dt \
                          + math.sqrt(max(volatility_value[j],0)*dt)*st_norm_draws[j,1])
    return [volatility_value, asset_price]

def european_put_option_pricing(theta, N, S_0, dt, er, M):
    sum_of_x = 0
    for i in range(1, M):
        sum_of_x += path(theta, N, S_0, dt, er)[1,N]
    return sum_of_x/M

abc = path(theta, 10, 100, 0.01, 0.05)[1,10]

print(abc)

import Heston_Calibration as hc
import numpy as np

theta = np.array([0.08, 0.10, -0.8, 3, 0.25])
#print(hc.eur_call_heston_price_GL(theta, 100, 5, 100, 0.05, 20, 10))

mkt_data = np.vstack(([hc.eur_call_heston_price_GL(theta, 80, 0.25, 100, 0.02, 100, 64), 80, 0.25],
                      [hc.eur_call_heston_price_GL(theta, 80, 0.5, 100, 0.02, 100, 64), 80, 0.5],
                      [hc.eur_call_heston_price_GL(theta, 80, 1, 100, 0.02, 100, 64), 80, 1],
                      [hc.eur_call_heston_price_GL(theta, 80, 2, 100, 0.02, 100, 64), 80, 2],
                      [hc.eur_call_heston_price_GL(theta, 90, 0.25, 100, 0.02, 100, 64), 90, 0.25],
                      [hc.eur_call_heston_price_GL(theta, 90, 0.5, 100, 0.02, 100, 64), 90, 0.5],
                      [hc.eur_call_heston_price_GL(theta, 90, 1, 100, 0.02, 100, 64), 90, 1],
                      [hc.eur_call_heston_price_GL(theta, 90, 2, 100, 0.02, 100, 64), 90, 2],
                      [hc.eur_call_heston_price_GL(theta, 100, 0.25, 100, 0.02, 100, 64), 100, 0.25],
                      [hc.eur_call_heston_price_GL(theta, 100, 0.5, 100, 0.02, 100, 64), 100, 0.5],
                      [hc.eur_call_heston_price_GL(theta, 100, 1, 100, 0.02, 100, 64), 100, 1],
                      [hc.eur_call_heston_price_GL(theta, 100, 2, 100, 0.02, 100, 64), 100, 2],
                      [hc.eur_call_heston_price_GL(theta, 110, 0.25, 100, 0.02, 100, 64), 110, 0.25],
                      [hc.eur_call_heston_price_GL(theta, 110, 0.5, 100, 0.02, 100, 64), 110, 0.5],
                      [hc.eur_call_heston_price_GL(theta, 110, 1, 100, 0.02, 100, 64), 110, 1],
                      [hc.eur_call_heston_price_GL(theta, 110, 2, 100, 0.02, 100, 64), 110, 2],
                      [hc.eur_call_heston_price_GL(theta, 120, 0.25, 100, 0.02, 100, 64), 120, 0.25],
                      [hc.eur_call_heston_price_GL(theta, 120, 0.5, 100, 0.02, 100, 64), 120, 0.5],
                      [hc.eur_call_heston_price_GL(theta, 120, 1, 100, 0.02, 100, 64), 120, 1],
                      [hc.eur_call_heston_price_GL(theta, 120, 2, 100, 0.02, 100, 64), 120, 2],
                      ))
#print(np.abs(mkt_data).max())

#x = hc.r_function(theta,mkt_data,100,0.05,20,10)
#print(x)
#print(np.dot(np.transpose(x),np.ones(7)))
#print(np.dot(mkt_data,np.ones(3)))
#print(hc.grad_heston_price_GL(theta, 5, 100, 100, 0.02, 20, 5))
#MATRIX = hc.J_matrix(theta,mkt_data,100,0.01,20,10)
#print(np.dot(np.transpose(MATRIX),MATRIX)) # takie cos daje macierz 5x5
#print(mkt_data)
theta_0 = np.array([0.07, 0.11, -0.7, 3.1, 0.28])
#print(hc.J_matrix(theta,mkt_data,100,0.02,100,64))
#print(hc.J_matrix(theta_0,mkt_data,100,0.02,100,10))

#print(hc.B_rho(theta,10,10))
print(hc.calibration_LM(theta_0,mkt_data,10,0.000001,0.000001,0.000001,100,0.02,50,10))


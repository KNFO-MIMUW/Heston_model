import numpy as np
import math as math
import cmath

# SECTION 1 - Functions, which are necessary to compute price of the european option for given parameters.
def ksi(theta, u):
    return theta[3] - theta[4]*theta[2]*u*1j
def d(theta, u):
    return cmath.sqrt(ksi(theta, u)*ksi(theta, u) + math.pow(theta[4], 2)*(u*u + u*1j))
def g1(theta, u):
    return (ksi(theta, u) + d(theta, u))/(ksi(theta, u) - d(theta, u))
def A1(theta, u, t):
    return (u*u + 1j*u)*cmath.sinh(d(theta, u)*t/2)
def A2(theta, u, t):
    return d(theta, u)*cmath.cosh(d(theta, u)*t/2)/theta[0] + ksi(theta, u)*cmath.sinh(d(theta, u)*t/2)/theta[0]
def A(theta, u, t):
    return A1(theta, u, t)/A2(theta, u, t)
def B(theta, u, t):
    return d(theta, u)*cmath.exp(theta[3]*t/2)/(A2(theta, u, t)*theta[0])
def D(theta, u, t):
    return cmath.log(d(theta, u)/theta[0]) + theta[3]*t/2 - cmath.log(A2(theta, u, t))
# Equation (18) p. 9 - characteristic function, which we are going to use in out project
def char_function(theta, u, t, S_0, r):
    return cmath.exp(1j*u*(math.log(S_0*math.exp(r*t)/S_0)) - t*theta[3]*theta[1]*theta[2]*1j*u/theta[4] - A(theta, u, t) + \
                     2*theta[3]*theta[1]*D(theta, u, t)/math.pow(theta[4], 2))
# integrate_char_function - integrals computed by means of Gauss-Legendre Quadrature
def integrate_char_function(theta, K, t, S_0, r, i, M, deg):
    x, w = np.polynomial.legendre.leggauss(deg)
    u = (x[0]+1)*0.5*M
    value = w[0]*cmath.exp(-1j*u*math.log(K/S_0))/(1j*u)*char_function(theta, u - i, t, S_0, r)
    for j in range(1, deg): # deg - number of nodes
        u = (x[j] + 1)*0.5*M
        value = value + w[j]*cmath.exp(-1j*u*math.log(K/S_0))/(1j*u)*char_function(theta, u - i, t, S_0, r)
    value = value*0.5*M
    return value.real
# eur_call_heston_price - Equation (9)
def eur_call_heston_price(theta, K, t, S_0, r, M, deg):
    return (S_0 - math.exp(-r*t)*K)/2 + math.exp(-r*t)/math.pi*(S_0*integrate_char_function(theta, K, t, S_0, r, 1j, M, deg) - \
                                                                K*integrate_char_function(theta, K, t, S_0, r, 0, M, deg))
# SECTION 2 - functions, which are necessary to compute gradient of characteristic function
def h_1(theta, u, t):
    return -A(theta, u, t)/theta[0]
def h_2(theta, u ,t):
    return 2*theta[3]*D(theta, u, t)/math.pow(theta[4], 2) - t*theta[3]*theta[2]*1j*u/theta[4]
def h_3(theta, u, t):
    return - A_rho(theta, u, t) + 2*theta[3]*theta[1]*(d_rho(theta,u) - d(theta, u)*A2_rho(theta, u, t)/A2(theta,u,t))/ \
                                  (theta[4]*theta[4]*d(theta, u)) - t*theta[3]*theta[1]*1j*u/theta[4]
def h_4(theta, u, t):
    return A_rho(theta, u, t)/(theta[4]*1j*u) + 2*theta[1]*D(theta, u, t)/(theta[4]*theta[4]) + \
           2*theta[3]*theta[1]*B_kappa(theta, u, t)/(theta[4]*theta[4]*B(theta, u, t)) - \
           t*theta[1]*theta[2]*1j*u/theta[4]
def h_5(theta, u, t):
    return - A_sigma(theta, u, t) - 4*theta[3]*theta[1]*D(theta, u, t)/(math.pow(theta[4], 3)) + \
           2*theta[3]*theta[1]*(d_rho(theta, u) - d(theta, u)*A2_sigma(theta, u, t)/A2(theta, u, t))/ \
           (theta[4]*theta[4]*d(theta, u)) + t*theta[3]*theta[1]*theta[2]*1j*u/(theta[4]*theta[4])
def d_rho(theta, u):
    return - ksi(theta, u)*theta[4]*1j*u/d(theta, u)
def A2_rho(theta, u, t):
    return - theta[4]*1j*u*(2 + t*ksi(theta, u))*(ksi(theta, u)*cmath.cosh(d(theta, u)*t/2) + \
                                                  d(theta, u)*cmath.sinh(d(theta, u)*t/2))/(2*d(theta, u)*theta[0])
def B_rho(theta, u, t):
    return cmath.exp(theta[3]*t/2)*(d_rho(theta, u)/A2(theta, u, t) - \
                                    A2_rho(theta, u, t)/(A2(theta,u,t)*A2(theta,u,t)))/theta[0]
def A1_rho(theta, u, t):
    return - 1j*u*(u*u + 1j*u)*t*ksi(theta, u)*theta[4]*cmath.cosh(d(theta, u)*t/2)/(2*d(theta, u))
def A_rho(theta, u, t):
    return A1_rho(theta, u, t)/A2(theta, u, t) - A2_rho(theta, u, t)*A(theta, u, t)/A2(theta, u, t)
def A_kappa(theta, u, t):
    return 1j*A_rho(theta, u, t)/(u*theta[4])
def B_kappa(theta, u, t):
    return B_rho(theta, u, t)*1j/(theta[4]*u) + t*B(theta, u, t)/2
def d_sigma(theta, u):
    return (theta[2]/theta[4] - 1/ksi(theta, u))*d_rho(theta, u) + theta[4]*u*u/d(theta, u)
def A1_sigma(theta, u, t):
    return (u*u + 1j*u)*t*d_sigma(theta, u)*cmath.cosh(d(theta, u)*t/2)/2
def A2_sigma(theta, u, t):
    return theta[2]*A2_rho(theta, u, t)/theta[4] - (2 + t*ksi(theta, u))*A1_rho(theta, u, t)/ \
                                                   (1j*u*t*ksi(theta, u)*theta[0]) + theta[4]*t*A1(theta, u, t)/(2*theta[0])
def A_sigma(theta, u, t):
    return A1_sigma(theta, u, t)/A2(theta, u, t) - A(theta, u, t)*A2_sigma(theta, u, t)/A2(theta, u, t)
def h(theta, u, t, which):
    if which == 1:
        return h_1(theta, u, t)
    if which == 2:
        return h_2(theta, u, t)
    if which == 3:
        return h_3(theta, u, t)
    if which == 4:
        return h_4(theta, u, t)
    if which == 5:
        return h_5(theta, u, t)
# integrate_grad_function - integrals computed by means of Gauss-Legendre Quadrature
def integrate_grad_function(theta, K, t, S_0, r, i, M, deg, which):
    x, w = np.polynomial.legendre.leggauss(deg)
    u = (x[0] + 1)*0.5*M
    value = w[0]*cmath.exp(-1j*u*math.log(K/S_0))/(1j*u)*char_function(theta, u - i, t, S_0, r)*h(theta, u - i, t, which)
    for j in range(1, deg):
            u = (x[j] + 1)*0.5*M
            value = value + w[j]*cmath.exp(-1j*u*math.log(K/S_0))/(1j*u)*char_function(theta, u - i, t, S_0, r)* \
                            h(theta, u - i, t, which)
    return value.real
# grad_heston_price - Equation (22)
def grad_heston_price(theta, t, K, S_0, r, M, deg):
    first_int = np.array(integrate_grad_function(theta, K, t, S_0, r, 1j, M, deg, 1))
    second_int = np.array(integrate_grad_function(theta, K, t, S_0, r, 0, M, deg, 1))
    for i in range(2, 6):
        first_int = np.append(first_int, integrate_grad_function(theta, K, t, S_0, r, 1j, M, deg, i))
        second_int = np.append(second_int, integrate_grad_function(theta, K, t, S_0, r, 0, M, deg, i))
    return (math.exp(-r*t)/math.pi)*(first_int - K*second_int)
def r_function(theta):
    r_vector = np.array(eur_call_heston_price(theta, mkt_data[0, 1], mkt_data[0, 2], S_0, r, M, deg) - \
                        mkt_data[0, 0])
    for i in range(1, len(mkt_data)):
        r_vector = np.append(r_vector, eur_call_heston_price(theta, mkt_data[i, 1], mkt_data[i, 2], S_0, r, M, deg) - \
                             mkt_data[i, 0])
    return r_vector





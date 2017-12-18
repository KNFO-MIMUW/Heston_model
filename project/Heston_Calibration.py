import cmath as cmath
import math as math
import numpy as np

# SECTION 1 - Functions, which are necessary to compute price of the european option for given parameters.
# Equation (18) p. 9 shows the characteristic function, which we are going to use in out project
def ksi(theta, u):
    return theta[3] - theta[4] * theta[2] * u * 1j
def d(theta, u):
    return cmath.sqrt(ksi(theta, u)*ksi(theta, u) + math.pow(theta[4], 2)*u*u + u*1j)
def g1(theta, u):
    return (ksi(theta, u) + d(theta, u))/(ksi(theta, u) - d(theta, u))
def A1(theta, u, t):
    return (u*u + 1j*u)*cmath.sinh(d(theta, u)*t/2)
def A2(theta, u, t):
    return d(theta, u)*cmath.cosh(d(theta, u)*t/2) + ksi(theta, u)*cmath.sinh(d(theta, u)*t/2)
def A(theta, u, t):
    return A1(theta, u, t)/A2(theta, u, t)
def B(theta, u, t):
    return d(theta, u)*cmath.exp(theta[3]*t/2)/A2(theta, u, t)
def D(theta, u, t):
    return cmath.log(d(theta, u)) + theta[3]*t/2 - cmath.log(A2(theta, u, t))
def char_function(theta, u, t, S_0, r):
    return cmath.exp(1j*u*(math.log(S_0) + r*t) - t*theta[3]*theta[1]*theta[2]*1j*u/theta[4] - theta[0]*A(theta, u, t) + \
                     2*theta[3]*theta[1]*D(theta, u, t)/math.pow(theta[4], 2))
def integrate_char_function(theta, K, t, S_0, r, i, M, n, rule):
    delta_t = M/n
    if rule == "Riemann":
        value = 0
        for j in range(1, n+1):
            value = value + cmath.exp(-1j*j*delta_t*math.log(K))/(1j*j*delta_t)*char_function(theta, j*delta_t-i, t, S_0, r)
        value = value*delta_t
    elif rule == "Simpson":
        value = cmath.exp(-1j*delta_t*math.log(K))/(1j*delta_t)*char_function(theta, delta_t-i, t, S_0, r)
        for j in range(2, n, 2):
            value = value + 4*cmath.exp(-1j*j*delta_t*math.log(K))/(1j*j*delta_t)*char_function(theta, j*delta_t - i, t, S_0, r)
        for j in range(3, n-1, 2):
            value = value + 2*cmath.exp(-1j*j*delta_t*math.log(K))/(1j*j*delta_t)*char_function(theta, j*delta_t - i, t, S_0, r)
        value = value + cmath.exp(-1j*M*math.log(K))/(1j*M)*char_function(theta, M - i, t, S_0, r)
        value = value*delta_t/3
    elif rule == "Gauss-Legendre":
        value = 0
        deg = 10
        #f = lambda x: cmath.exp(-1j*x*math.log(K))/(1j*x)*char_function(theta,x-i,t,S_0,r)
        x, w = np.polynomial.legendre.leggauss(deg)
        for j in range(0, deg): # Tutaj siedzie deg
            u = (x[j] + 1)*0.5*M
            value = value + w[j]*cmath.exp(-1j*u*math.log(K))/(1j*u)*char_function(theta, u - i, t, S_0, r)
        value = value*0.5*M
    return value.real
def integrate_char_function_GL(theta, K, t, S_0, r, i, M, deg):
    x, w = np.polynomial.legendre.leggauss(deg)
    u = (x[0]+1)*0.5*M
    value = w[0]*cmath.exp(-1j*u*math.log(K))/(1j*u)*char_function(theta, u - i, t, S_0, r)
    for j in range(1, deg): # Tutaj siedzi deg
        u = (x[j] + 1)*0.5*M # Nie musialobyc tej linijki, tutaj jest jakis problem z formatem
        value = value + w[j]*cmath.exp(-1j*u*math.log(K))/(1j*u)*char_function(theta, u - i, t, S_0, r)
    value = value*0.5*M
    return value.real
def eur_call_heston_price(theta, K, t, S_0, r, M, n, rule):
    return (S_0 - math.exp(-r*t)*K)/2 + math.exp(-r*t)/math.pi*(integrate_char_function(theta, K, t, S_0, r, 1j, M, n, rule) - \
                                                                K*integrate_char_function(theta, K, t, S_0, r, 0, M, n, rule))
def eur_call_heston_price_GL(theta, K, t, S_0, r, M, deg):
    return (S_0 - math.exp(-r*t)*K)/2 + math.exp(-r*t)/math.pi*(integrate_char_function_GL(theta, K, t, S_0, r, 1j, M, deg) - \
                                                                K*integrate_char_function_GL(theta, K, t, S_0, r, 0, M, deg))
#theta = np.array([0.09, 0.09, -0.3, 2, 1])
#print(eur_call_heston_price(theta, 100, 5, 100, 0.05, 10, 10000, "Riemann"))
#print(eur_call_heston_price(theta, 100, 5, 100, 0.05, 10, 10000, "Simpson"))
#print(eur_call_heston_price(theta, 100, 5, 100, 0.05, 10, 10000, "Gauss-Legendre"))
#print(eur_call_heston_price_GL(theta, 100, 5, 100, 0.05, 10, 10))

# SECTION 2 - Functions, which are necessary to compute gradient of the characteristic function.
# Equation (22) p. 11 shows the gradient of the characteristic function, which we are going to use in out project
def h_1(theta, u, t):
    return -A(theta,u,t)
def h_2(theta, u ,t):
    return 2*theta[3]*D(theta,u,t)/math.pow(theta[4],2)-t*theta[3]*theta[2]*1j*u/theta[4]
def h_3(theta, u, t):
    return -theta[0]*A_rho(theta,u,t)+2*theta[3]*theta[1]*(d_rho(theta,u)-d(theta,u)*A2_rho(theta,u,t)/A2(theta,u,t))/(theta[4]*theta[4]*B(theta,u,t))-t*theta[3]*theta[1]*1j*u/theta[4]
def h_4(theta, u, t):
    return theta[0]*A_rho(theta,u,t)/(theta[4]*1j*u)+2*theta[1]*D(theta,u,t)/(theta[4]*theta[4])+2*theta[3]*theta[1]*B_kappa(theta,u,t)/(theta[4]*theta[4]*B(theta,u,t))-t*theta[1]*theta[2]*1j*u/theta[4]
def h_5(theta, u, t):
    return -theta[0]*A_sigma(theta,u,t)-4*theta[3]*theta[1]*D(theta,u,t)/(math.pow(theta[4],3))+2*theta[3]*theta[1]*(d_rho(theta,u)-d(theta,u)*A2_sigma(theta,u,t)/A2(theta,u,t))/(theta[4]*theta[4]*d(theta,u))+t*theta[3]*theta[1]*theta[2]*1j*u/(theta[4]*theta[4])
def d_rho(theta, u):
    return -ksi(theta,u)*theta[4]*1j*u/d(theta,u)
def A2_rho(theta, u, t):
    return -theta[4]*1j*u*(2+t*ksi(theta,u))*(ksi(theta,u)*cmath.cosh(d(theta,u)*t/2)+d(theta,u)*cmath.sinh(d(theta,u)*t/2))/(2*d(theta,u))
def B_rho(theta, u, t):
    return cmath.exp(theta[3]*t/2)*(d_rho(theta, u)/A2(theta,u,t)-A2_rho(theta,u,t)/(A2(theta,u,t)*A2(theta,u,t)))
def A1_rho(theta, u, t):
    return -1j*u*(u*u+1j*u)*t*ksi(theta,u)*theta[4]*cmath.cosh(d(theta,u)*t/2)/(2*d(theta,u))
def A_rho(theta, u, t):
    return A1_rho(theta,u,t)/A2(theta,u,t)-A2_rho(theta,u,t)*A(theta,u,t)/A2(theta,u,t)
def A_kappa(theta, u, t):
    return 1j*A_rho(theta,u,t)/(u*theta[4])
def B_kappa(theta, u, t):
    return B_rho(theta,u,t)*1j/(theta[4]*u)+t*B(theta,u,t)/2
def d_sigma(theta, u):
    return (theta[2]/theta[4]-1/ksi(theta,u))*d_rho(theta,u)+theta[4]*u*u/d(theta,u)
def A1_sigma(theta, u, t):
    return (u*u+1j*u)*t*d_sigma(theta, u)*cmath.cosh(d(theta,u)*t/2)/2
def A2_sigma(theta, u, t):
    return theta[2]*A2_rho(theta,u,t)/theta[4]-(2+t*ksi(theta,u))*A1_rho(theta,u,t)/(1j*u*t*ksi(theta,u))+theta[4]*t*A1(theta,u,t)/2
def A_sigma(theta, u, t):
    return A1_sigma(theta,u,t)/A2(theta,u,t)-A(theta,u,t)*A2_sigma(theta,u,t)/A2(theta,u,t)
def h(theta, u, t, which):
    if which == 1:
        return h_1(theta,u,t)
    if which == 2:
        return h_2(theta,u,t)
    if which == 3:
        return h_3(theta,u,t)
    if which == 4:
        return h_4(theta,u,t)
    if which == 5:
        return h_5(theta,u,t)

def J_matrix(theta, mkt_data, S_0, r, M, deg):
    J = grad_heston_price_GL(theta,mkt_data[0,2],mkt_data[0,1],S_0,r,M,deg)
    J.shape = (5,1)
    for i in range(1,len(mkt_data)):
        J_next = grad_heston_price_GL(theta,mkt_data[i,2],mkt_data[i,1],S_0,r,M,deg)
        J_next.shape = (5,1)
        J = np.hstack((J,J_next))
    return J
def integrate_grad_function_GL(theta, K, t, S_0, r, i, M, deg, which):
    x, w = np.polynomial.legendre.leggauss(deg)
    u = (x[0]+1)*0.5*M
    value = w[0]*cmath.exp(-1j*u*math.log(K))/(1j*u)*char_function(theta,u-i,t,S_0,r)*h(theta,u-i,t,which)
    for j in range(1,deg):
        u = (x[j]+1)*0.5*M
        value = value + w[j]*cmath.exp(-1j*u*math.log(K))/(1j*u)*char_function(theta,u-i,t,S_0,r)*h(theta,u-i,t,which)
    return value.real
def grad_heston_price_GL(theta, t, K, S_0, r, M, deg):
    first_int = np.array(integrate_grad_function_GL(theta,K,t,S_0,r,1j,M,deg,1))
    second_int = np.array(integrate_grad_function_GL(theta,K,t,S_0,r,0,M,deg,1))
    for i in range(2,6):
        first_int = np.append(first_int,integrate_grad_function_GL(theta,K,t,S_0,r,1j,M,deg,i))
        second_int = np.append(second_int,integrate_grad_function_GL(theta,K,t,S_0,r,0,M,deg,i))
    return (math.exp(-r*t)/math.pi)*(first_int-K*second_int)
def r_function(theta, mkt_data, S_0, r, M, deg):
    r_vector = np.array(eur_call_heston_price_GL(theta,mkt_data[0,1],mkt_data[0,2],S_0,r,M,deg) - mkt_data[0,0])
    for i in range(1,len(mkt_data)):
        r_vector = np.append(r_vector,eur_call_heston_price_GL(theta, mkt_data[i,1], mkt_data[i,2], S_0, r, M, deg) - mkt_data[i,0])
    return r_vector
def norm_r(theta, mkt_data, S_0, r, M, deg):
    r = r_function(theta,mkt_data,S_0,r,M,deg)
    return math.sqrt(np.dot(np.transpose(r),r))
def delta_theta_function(theta, mkt_data, S_0, r, M, deg, mu):
    J = J_matrix(theta,mkt_data,S_0,r,M,deg)
    return np.linalg.solve(np.dot(J,np.transpose(J))+mu*np.identity(5),np.dot(J,r_function(theta,mkt_data,S_0,r,M,deg)))


# Levenberg-Marquardt algorithm to make heston model calibration
def calibration_LM(theta, mkt_data, tau, eps1, eps2, eps3, S_0, r, M, deg):
    k=0
    J = J_matrix(theta,mkt_data,S_0,r,M,deg)
    norm_r_theta = norm_r(theta,mkt_data,S_0,r,M,deg)
    mu = tau*np.dot(np.transpose(J),J).diagonal().max() # tutaj nie wiem do konca czym jest diag(J), gdy macierz nie jest kwadratowa
    v = 2
    while 1==1:
        k=k+1
        delta_theta = -delta_theta_function(theta,mkt_data,S_0,r,M,deg,mu)
        theta_next = theta + delta_theta
        norm_r_theta_next = norm_r(theta_next,mkt_data,S_0,r,M,deg)
        delta_L = np.dot(np.transpose(delta_theta),mu*delta_theta+np.dot(J,r_function(theta,mkt_data,S_0,r,M,deg)))
        print(norm_r_theta)
        delta_F = norm_r_theta - norm_r_theta_next
        norm_r_theta = norm_r_theta_next
        theta = theta_next
        if (delta_L > 0 and delta_F > 0): # Wyznaczyc funcje na delta_L oraz delta_F
            J = J_matrix(theta,mkt_data,S_0,r,M,deg)
            if (norm_r_theta <= eps1 or np.dot(np.transpose(J),np.ones(5)).max() <= eps2 or np.dot(np.transpose(delta_theta),delta_theta)/np.dot(np.transpose(theta),theta) <= eps3):
                return theta
        else:
            mu = mu*v
            v = 2*v
    return theta

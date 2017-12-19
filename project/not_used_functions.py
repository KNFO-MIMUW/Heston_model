# Set of functions, which are not used in the final version of the project
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
def eur_call_heston_price(theta, K, t, S_0, r, M, n, rule):
    return (S_0 - math.exp(-r*t)*K)/2 + math.exp(-r*t)/math.pi*(integrate_char_function(theta, K, t, S_0, r, 1j, M, n, rule) - \
                                                                K*integrate_char_function(theta, K, t, S_0, r, 0, M, n, rule))
def J_matrix(theta, mkt_data, S_0, r, M, deg):
    J = grad_heston_price_GL(theta,mkt_data[0,2],mkt_data[0,1],S_0,r,M,deg)
    J.shape = (5,1)
    for i in range(1,len(mkt_data)):
        J_next = grad_heston_price_GL(theta,mkt_data[i,2],mkt_data[i,1],S_0,r,M,deg)
        J_next.shape = (5,1)
        J = np.hstack((J,J_next))
    return J
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
# calibration_LM - issues with convergence
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

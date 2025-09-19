import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import place_poles
# study stability of Stanley controller

# copied from f1tenth_gym/gym/f110_gym/envs/f110_env.py
params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
m = params['m'] # mass of vehicle
Csf = params['C_Sf']
Csr = params['C_Sr']
lf = params['lf']
lr = params['lr']
Iz = params['I']
g = 9.81
mu = params['mu']
h = params['h']

def ackermann_placement(A, B, poles):
    n = A.shape[0]
    # controllability mtx
    C = np.hstack([B] + [np.linalg.matrix_power(A, i) @ B for i in range(1, n)])
    rank = np.linalg.matrix_rank(C)
    if rank != n:
        raise ValueError('system not controllable')
    p = np.real(np.poly(poles))  # Coefficients of desired characteristic polynomial
    # Place the poles using Ackermann's method
    n = np.size(p)
    pmat = p[n-1] * np.linalg.matrix_power(A, 0)
    for i in np.arange(1, n):
        pmat = pmat + p[n-i-1] * np.linalg.matrix_power(A, i)
    K = np.linalg.solve(C, pmat)
    K = K[-1][:]
    return K

# E = [e_y, de_y+dt, e_phi, de_phi_dt]
# dE_dt = A @ E + B @ steering + C @ dphi_dt_target
vx_vec = np.linspace(1.0,6.0,600)
#vx_vec = [1.0]
K_vec = []
for vx in vx_vec:
    ax = 0.0
    # "convert" load transfer to Caf
    Caf = m*mu*Csf/(lr+lf) *(g*lr-ax*h)
    Car = m*mu*Csr/(lr+lf) *(g*lf+ax*h)

    # dynamics according to textbook, without load transfer
    A = np.array([ [0,1,0,0],\
            [0, - (Caf + Car)/(m*vx), (Caf + Car)/m, (-Caf*lf + Car*lr)/(m*vx)], \
            [0, 0, 0, 1],\
            [0, -(Caf*lf - Car*lr)/(Iz*vx), (Caf*lf-Car*lr)/Iz, -(Caf*lf*lf + Car*lr*lr)/(Iz*vx)]
            ])

    B = np.array([[0, Caf/m, 0, Caf*lf/Iz]]).T
    C = np.array([[0, -(Caf*lf - Car*lr)/(m*vx) - vx, 0, -(Caf*lf*lf + Car*lr*lr)/(Iz*vx)]]).T

    print(f'vx = {vx}')
    print('eigenvalues of A')
    rank = np.linalg.matrix_rank(A)
    print(f'rank(A) {rank}')
    eigs = np.linalg.eig(A)
    for val in eigs[0]:
        print(f'{val.real:6.2f}, {val.imag:6.2f}i')
    '''
    print('eigen vectors of A')
    for vec in eigs[1]:
        for val in vec:
            print(f'{val.real:6.2f}, {val.imag:6.2f}i')
        print('--')
    '''

    # place_poles does not work when B.shape[1] == 1
    '''
    full_state_feedback = place_poles(A,B, [-1,-2,-3,-4])
    print('gain matrix:',full_state_feedback.gain_matrix)
    print('computed poles', full_state_feedback.computed_poles)
    print('iterations:', full_state_feedback.nb_iter)
    '''
    K = ackermann_placement(A,B,[-4,-3,-2,-1]).reshape(1,-1)
    K = np.clip(K,0,10)
    #K = np.array([[0.1,0.0,0.1,0.1]])
    print('K: ', K)
    print('closed loop eigenvals')
    eigs = np.linalg.eig(A - B @ K)
    for val in eigs[0]:
        print(f'{val.real:6.2f}, {val.imag:6.2f}i')
    K_vec.append(K)

K_vec = np.array(K_vec)[:,0,:]
with open('stanley_gains.p','wb') as f:
    pickle.dump(K_vec,f)
plt.plot(vx_vec, K_vec[:,0], '-',label='0')
plt.plot(vx_vec, K_vec[:,1], '-',label='1')
plt.plot(vx_vec, K_vec[:,2], '-',label='2')
plt.plot(vx_vec, K_vec[:,3], '-',label='3')

p0 = np.polyfit(vx_vec, K_vec[:,0], deg=5)
p1 = np.polyfit(vx_vec, K_vec[:,1], deg=5)
p2 = np.polyfit(vx_vec, K_vec[:,2], deg=5)
p3 = np.polyfit(vx_vec, K_vec[:,3], deg=5)

'''
vv = np.linspace(1,6)
plt.plot(vv, np.polyval(p0, vv),'--')
plt.plot(vv, np.polyval(p1, vv),'--')
plt.plot(vv, np.polyval(p2, vv),'--')
plt.plot(vv, np.polyval(p3, vv),'--')
'''
plt.legend()
plt.show()

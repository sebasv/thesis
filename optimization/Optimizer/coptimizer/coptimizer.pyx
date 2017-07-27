import numpy as np
cimport numpy as np
from scipy import optimize
import pandas as pd


cdef _error(np.ndarray[double] M, np.ndarray[double]  S, float k, float C):
    # return np.max(0, C - np.mean(Mu*np.maximum(0, S-k))) + np.max(0, np.mean(Ml*np.maximum(0, S-k)) - C)
    return np.square(0, C - np.mean(M*np.maximum(0, S-k)))


cdef squared_error(np.ndarray[double] T, np.ndarray[double] K, np.ndarray[double] C, np.ndarray[double, ndim=2] M, np.ndarray[double, ndim=2] S, float dt):
    cdef float err = 0
    cdef float t, k ,c
    cdef int idx
    for i in range(len(T)):
        t = T[i]
        k = K[i]
        c = C[i]
        idx = int(t/dt)
        err += _error(M[idx], S[idx], k, c)
    return err


cdef cheyette(float kappa, float sigma, int n, int p, np.ndarray[double, ndim=2] W, float dt, float f):
    cdef np.ndarray[double, ndim=2] X = np.zeros((n+1, p), dtype=np.float64)
    cdef float Y = 0
    for i in range(n):
        for j in range(p):
            X[i+1, j] = X[i, j] + (Y -kappa*X[i, j])*dt + sigma*W[i, j]
        Y += (sigma**2-2*kappa*Y)*dt
    return X + f


cdef CIR(np.ndarray[double, ndim=2] W, int n, int p, float dt, float x0, float kappa, float xi, float sigma):
    """CIR process from randomness W, initial value x0 and parameters"""
    cdef np.ndarray[double, ndim=2] X = np.ones((n+1, p), dtype=np.float64)*x0
    for i in range(n):
        for j in range(p):
            X[i+1, j] = np.abs( X[i, j] + kappa*(xi - X[i, j])*dt + np.sqrt(X[i, j])*sigma*W[i, j] )
    return X


cdef estimate(np.ndarray[double, ndim=2] V, np.ndarray[double, ndim=3] W, np.ndarray[double, ndim=2] r, float dt, float qS, float delta, float sS):
    cdef float beta = -qS / (sS * sS) / V
    cdef np.ndarray[double, ndim=2] S = np.cumprod(1 + (r+qS)*dt + (W@np.r_[0, sS, 0]) * V, 0)    
    cdef np.ndarray[double, ndim=2] M = np.cumprod(1 - r     *dt + W @ np.r_[delta, sS*beta, 0], 0)
    return S, M


def optimize( W,  r, float dt,  T, K, C):
    params = np.zeros(3)
    args = np.zeros(3)
    opt = optimize.minimize(
        optim_func_without_v,
        args,
        args=(params, T, K, C, dt, W, r),
        method='Powell'
    )
    return args, params


def optim_func_without_v(args, params, T, K, C, dt, W, r):
    V = 1 + np.sqrt( CIR(W[:, :, 0], W.shape[0], W.shape[1], dt, args[0], args[1], args[0], args[2])[:-1, :] )
    params_, fun = optimize_given_v(T, K, C, V, W, r, dt, params)
    params -= params
    params += params_
    return fun


cdef optim_func(np.ndarray[double] args, np.ndarray[double] T, np.ndarray[double] K, np.ndarray[double] C, float dt, np.ndarray[double, ndim=2] V, np.ndarray[double, ndim=3] W, np.ndarray[double, ndim=2] r):
    S, M = estimate(V, W, r, dt, args[0], args[1], args[2])
    return squared_error(T, K, C, M, S, dt)
    

#cdef calculate( np.ndarray[double] bla, ndim=2]V, np.ndarray[double, ndim=3] W, np.ndarray[double, ndim=2] r, float dt, float qS, float delta, float sS, float gamma, float sV):
#    cdef float beta = -qS / (sS * sS) / V
#    cdef np.ndarray[double, ndim=2] S = np.cumprod(1 + (r+qS)*dt + (W@np.r_[0, sS, sV]) * V, 0)
#    cdef np.ndarray[double, ndim=2] lu = W @ np.r_[0, 0, gamma]
#    cdef np.ndarray[double, ndim=2] Mu = np.cumprod(1 -  r    *dt + W @ np.r_[delta, sS*beta, 0] + lu, 0)
#    cdef np.ndarray[double, ndim=2] Ml = np.cumprod(1 -  r    *dt + W @ np.r_[delta, sS*beta, 0] - lu, 0)
#    return S, Mu, Ml
    # cdef np.ndarray[double] SMu = np.mean(S*Mu, 1)
    # cdef np.ndarray[double] SMl = np.mean(S*Ml, 1)
    # return SMu, SMl


def optimize_given_v( T,  K,  C,  V,  W, r,  dt, params):
    opt = optimize.minimize(
        optim_func,
        params,
        args=(T, K, C, V, W, r, dt),
        method='Powell'
    )
    return opt.x, opt.fun
# TODO optimization routine for given V
# TODO optimization routine for V
# TODO simulation routine


class Calibration(object):
    snu = np.array([1,0,0])
    sV  = np.array([0,1,0])

    def __init__(self, r_xi, r_kappa, r_f, con):
        self.r_params = [r_xi, r_kappa, r_f]
        self.v_params = []
        
        #df = pd.read_sql('select * from option_data', con, index_col='index')
        #df.columns = df.columns.astype(float)
        #self.data = df.sort_index(0).sort_index(1)
        
    def update_V(self, kappa_v, xi_v, sigma_v):
        if not self.v_params == [kappa_v, xi_v, sigma_v]:
            self.v_params = [kappa_v, xi_v, sigma_v]
            self.V = 1 + np.sqrt( CIR(xi_v, self.W[:,:,0], self.discretization_steps, self.n_simulations, self.dt, kappa_v, xi_v, sigma_v)[:-1] )
        
    def calculate(self, qS, kappa_v, xi_v, sigma_v, delta, sS, gamma, sV):
        self.update_V(kappa_v, xi_v, sigma_v)
        V = self.V
        beta = -qS / np.inner(sS, sS) / V
        S = np.cumprod(1 + (self.r+qS)*self.dt + (self.W@(sS + sV)) * V, 0)
        p3 = beta * (self.W @ sS) + self.W @ (delta*self.snu)
        lu = self.W @ (gamma * sV)
        Mu = np.cumprod(1 -  self.r    *self.dt + p3 + lu, 0)
        Ml = np.cumprod(1 -  self.r    *self.dt + p3 - lu, 0)
        return S, Mu, Ml, V

    def set_simulation(self, dt=.01, n_simulations=100, T=10):
        if T < self.data.columns.max():
            raise ValueError("max maturity too small for data")
        self.discretization_steps = int(T/dt)+1
        self.n_simulations = n_simulations
        self.dt = dt
        self.W = np.random.normal(scale=np.sqrt(self.dt), size=(self.discretization_steps, n_simulations, 3))
        self.Wr = np.random.normal(scale=np.sqrt(self.dt), size=(self.discretization_steps-1, n_simulations))
        #self.r = cheyette(*self.r_params[:2], *self.Wr.shape, self.Wr, self.dt, self.r_params[2])
        self.v_params = []

    def simulate(self, args):
        return self.calculate(*args[:5], np.r_[args[5:8]], args[8], args[9])
        
    
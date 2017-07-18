import sqlite3
import pandas as pd
import numpy as np

class Optimizer(object):
    con = sqlite3.connect(r'D:\data\scriptie\optimization.sqlite')
    df = pd.read_csv(r'd:\data\scriptie\spx_option_quotes.csv')


def estimate_short_rate():
    """
    Returns a CIR representation of the zero-coupon data from WRDS.
    """
    zyc = pd.read_csv(r'd:\data\scriptie\zero_coupon_yield_curve.csv')
    short_rate = zyc.sort_values(['date','days']).groupby('date').first()['rate'].div(100)

    dt = 1/365
    Y = short_rate.diff().shift(-1).div(np.sqrt(short_rate*dt)).values
    X = c_[1 / np.sqrt(short_rate.values), -np.sqrt(short_rate.values)] * np.sqrt(dt) 
    (a, rkappa), resid = np.linalg.lstsq(X[:-1], Y[:-1])[:2]
    rs = np.sqrt(resid/len(Y)).item()
    rxi = a/b
    return rs, rxi, rkappa # (0.05555211960687775, 0.001819691482190981, 0.26823591027620042)


def simulate(Sh=_Sh, qS=_qS, qV=_qV, sS=_sS, sV=_sV, r=_r, k=_k):
    gamma_lb = - np.sqrt(Sh**2 - (qS/sS)**2)
    gamma_ub = + np.sqrt(Sh**2 - (qS/sS)**2)
    
    sigmaM_lb = np.array([qS/sS, gamma_lb])
    muM_lb = -.5 * sigmaM_lb @ sigmaM_lb.T

    sigmaM_ub = np.array([qS/sS, gamma_ub])
    muM_ub = -.5 * sigmaM_ub @ sigmaM_ub.T

    sigmaM = np.array([qS/sS, 0])
    muM = -.5 * sigmaM @ sigmaM.T

    sigmaV = np.array([sV*r, sV*bar(1,r)])
    muV = -qV -.5 * sigmaV @ sigmaV.T
    
    V = np.exp(muV + sigmaV @ W)
    M_lb = np.exp(muM_lb + sigmaM_lb @ W)
    M = np.exp(muM + sigmaM @ W)
    M_ub = np.exp(muM_ub + sigmaM_ub @ W)
    F = np.maximum(0, V-k)
    return V, M_lb, M, M_ub, F


def CIR(x0, W, n, dt, kappa, xi, sigma):
    """CIR process from randomness W, initial value x0 and parameters"""
    X = np.ones(n+1)*x0
    for i in range(n):
        X[i+1] = abs( X[i] + kappa*(xi - X[i])*dt + np.sqrt(X[i])*sigma*W[i] )
    return X

def geom(x0, W, n, dt, R, V, q, sigma):
    """Geometric process from randomness W, initial value x0 and parameters"""
    X = cumprod(1 + (R+q)*dt + V*sigma@W)
    return r_[1,X]

class Simulation(object):
    self.snu = np.array([0,0,1])

    def __init__(self, step_per_t=100, n_simulations=1000, T=1):
        self.discretization_steps = int(step_per_t*T)
        self.n_simulations = n_simulations
        self.T = T
        self.dt = 1/step_per_t
        self.W = np.random.normal(scale=np.sqrt(self.dt), size=(n_simulations, self.discretization_steps, 3))
        self.Wr = np.random.normal(scale=np.sqrt(self.dt), size=(n_simulations, self.discretization_steps))

    def set_interest(self, rf0, kappa, xi, sigma):
        self.r = np.c_[[CIR(rf0, W, self.discretization_steps, self.dt, kappa, xi, sigma) for W in self.Wr]]

    def calculate(self, qS, sS, sV, delta):
        V = np.c_[[1 + np.sqrt( CIR(v0, W, self.discretization_steps, self.dt, kappa_v, xi_v, sigma_v) ) for W in self.W]]
        beta = -qS / (np.inner(sS, sS)*V)
        S = np.c_[[np.prod(1 + (r+qS)*self.dt + (                    sV @ W.T) * v) for W,v,r in zip(self.W, V, self.r)]]
        M = np.c_[[np.prod(1 + -r    *self.dt + (b*sS + delta*self.snu) @ W.T     ) for W,b,r in zip(self.W, beta, self.r)]] # no price bound so gamma == 0
        return S, M

    def calculate_call(self, qS, sS, sV, delta, K):
        S, M = self.calculate(qS, sS, sV, delta)
        return np.mean(M*np.maximum(0, S-K))

    def calculate_efficient(self, qS, sS, sV, delta, kappa_v, xi_v, sigma_v):
        """returns the efficient market option price for the given parameters"""
        V = 1 + np.c_[[np.sqrt( CIR(v0, W, self.discretization_steps, self.dt, kappa_v, xi_v, sigma_v) ) for W in self.W]]
        beta = -qS / (np.inner(sS, sS)*V)
        return np.c_[[np.prod( 1 + W@((1 + b)*sS + delta*self.snu) ) for W,b in zip(self.W, beta)]]
        
        


    if __name__ == '__main__':
        df = pd.read_csv(r'd:\data\scriptie\spx_option_quotes.csv')

    
        N = int(1e5)
        W = np.random.normal(scale=1/np.sqrt(N), size=(N,))
        x = CIR(0, W, N, 1/N, rkappa, rxi, rs)


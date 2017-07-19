import sqlite3
import pandas as pd
import numpy as np
import json
from scipy import stats, optimize
from matplotlib import pyplot as plt

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


def cheyette(kappa, sigma, n, p, W, dt, f):
    X = np.zeros((n+1, p))
    Y = 0
    for i in range(n):
        X[i+1] = X[i] + (Y -kappa*X[i])*dt + sigma*W[i]
        Y     += (sigma**2-2*kappa*Y)*dt
    return X + f

def CIR(x0, W, n, p, dt, kappa, xi, sigma):
    """CIR process from randomness W, initial value x0 and parameters"""
    X = np.ones((n+1, p))*x0
    for i in range(n):
        X[i+1] = np.abs( X[i] + kappa*(xi - X[i])*dt + np.sqrt(X[i])*sigma*W[i] )
    return X


def iv_to_call(df):
    d1 = lambda s, k, ss, t: np.log(s/k)/ss/np.sqrt(t) + ss*np.sqrt(t)
    d2 = lambda s, k, ss, t: d1(s,k, ss, t) - ss*np.sqrt(t)
    bs = lambda s, k, ss, t: stats.norm.cdf(d1(s, k, ss, t))*s - stats.norm.cdf(d2(s, k, ss, t))*k
    data = df.copy()
    for t in data.columns:
        for k in data.index:
            data.loc[k, t] = bs(1, k, data.loc[k, t]/100, t) #impvol to BS price
    return data


class CheyetteCalibration(object):
    def __init__(self, data):
        self.data = iv_to_call(data)

    def calibrate(self, p=100, dt=.01, f=0.02, T=10, con=None):
        self.f = f
        self.dt = dt
        if T < self.data.columns.max():
            raise ValueError("max maturity too small for data")
        n = int(T/dt)+1
        self.W = W = np.random.normal(scale=np.sqrt(dt), size=(n, p))
        def fun(params):
            r = cheyette(*params, n, p, W, dt, f)
            fit = np.sum(np.square([
                np.mean(np.maximum(r[int(t/dt)]-k, 0)) - self.data.loc[k, t] 
                if not np.isnan(self.data.loc[k, t])
                else 0
                for t in self.data.columns for k in self.data.index
            ]))
            print('got fit ', fit)
            return fit
        self.x = optimize.minimize(fun, np.random.exponential(1,2), method='nelder-mead') 
        if con is not None:
            con.execute(
                '''insert into cheyette(kappa, xi, p, dt, f, fun)
                values (?,?,?,?,?,?)''', [*self.x.x, p, dt, f, self.x.fun])
            con.commit()
        return self.x

    def simulate(self):
        return cheyette(*self.x.x, *self.W.shape, self.W, self.dt, self.f)


class Calibration(object):
    snu = np.array([1,0,0])
    sV  = np.array([0,1,0])

    def __init__(self, r_xi, r_kappa, r_f, data):
        self.r_params = [r_xi, r_kappa, r_f]
        self.data = iv_to_call(data)
        
    def calculate(self, qS, kappa_v, xi_v, sigma_v, delta, sS):
        V = 1 + np.sqrt( CIR(xi_v, self.W[:,:,0], self.discretization_steps, self.n_simulations, self.dt, kappa_v, xi_v, sigma_v)[:-1] )
        beta = -qS / np.inner(sS, sS)
        S = np.cumprod(1 + (self.r+qS)*self.dt + (self.W @ self.sV) * V , 0)
        p1 = beta * (self.W @ sS) / V
        p2 = self.W @ (delta*self.snu)
        p3 = p1 + p2
        M = np.cumprod(1 -  self.r    *self.dt + p3, 0)
        return S, M

    def calibrate(self, dt=.01, n_simulations=100, T=10, con=None):
        if T < self.data.columns.max():
            raise ValueError("max maturity too small for data")
        self.discretization_steps = int(T/dt)+1
        self.n_simulations = n_simulations
        self.dt = dt
        self.W = np.random.normal(scale=np.sqrt(self.dt), size=(self.discretization_steps, n_simulations, 3))
        self.Wr = np.random.normal(scale=np.sqrt(self.dt), size=(self.discretization_steps-1, n_simulations))
        self.r = cheyette(*self.r_params[:2], *self.Wr.shape, self.Wr, self.dt, self.r_params[2])

        def fun(args):
            S, M = self.calculate(*args[:5], np.r_[args[5:8]])
            fit = np.sum(np.square([
                self.data.loc[k, t] - np.mean(M[int(t/dt)]*np.maximum(0, S[int(t/dt)]-k))
                if not np.isnan(self.data.loc[k, t])
                else 0
                for k in self.data.index for t in self.data.columns
            ]))
            print('got fit ', fit)
            return fit

        self.x = optimize.minimize(fun, np.random.exponential(1, 8), method='nelder-mead') 
        if con is not None:
            con.execute(
                '''insert into heston(qS, kappa_v, xi_v, sigma_v, delta, sS1, sS2, sS3, p, dt, kappa_r, xi_r, f, fun)
                values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', [*self.x.x, self.n_simulations, self.dt, *self.r_params, self.x.fun])
            con.commit()
        return self.x


if __name__ == '__main__':
    con = sqlite3.connect('results.sqlite')
    con.execute('create table if not exists cheyette(kappa real, xi real, created datetime default current_timestamp, p real, dt real, f real, fun real)')
    con.execute('create table if not exists heston(qS real, kappa_v real, xi_v real, sigma_v real, delta real, sS1 real, sS2 real, sS3 real, created datetime default current_timestamp, p real, dt real, kappa_r real, xi_r real, f real, fun real)')
    forward_curve = .002

    print('started')
    with open('usd_swaption_data.json') as f:
        js = json.load(f)
    df = pd.Panel(js).loc[:,'log-normal vol',:]
    df.index = df.index.str.replace('ATM','0.00%').str.replace('%','').astype('float')/100 + 1
    df.columns = df.columns.astype(int)
    print('initializing')
    cc = CheyetteCalibration(df)
    print('started optimization')
    print(cc.calibrate(p=10000, f=forward_curve, con=con)) #  0.9912468 ,  1.29428977

    #plt.plot(cheyette(*cc.x.x, 100, 10, np.random.normal(0, .1, (100,10)), .01, forward_curve))
    #plt.plot(np.mean(cheyette(*cc.x.x, 100, 10, np.random.normal(0, .1, (100,10)), .01, forward_curve),1))
    
    with open('spx_option_data.json') as f:
        js = json.load(f)
    df = pd.Panel(js).loc[:,:,'2017/06/30']
    df.columns = df.columns.astype(int)/10
    df.index = pd.to_datetime(df.index, format='%Y/%m/%d') - pd.to_datetime('2017/06/30', format='%Y/%m/%d')
    df.index = df.index.days/365
    df = df[df.index>0]
    df /= 100
    ca = Calibration(*cc.x.x, forward_curve, df.T)
    print(ca.calibrate(n_simulations=10000, con=con))



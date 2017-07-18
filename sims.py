# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 12:02:23 2017

@author: mail
"""
import numpy as np
import pandas as pd
import sqlite3
from scipy.optimize import minimize

if 'bla' not in globals():
    with sqlite3.connect(r'd:\data\scriptie\wrds.sqlite') as con:
        bla = pd.read_sql(
            """select strike_price/close strike, (best_bid+best_offer)/2/close option, rate rf, maturity/365. t
            from spx where date(observation) between date('2001-03-29') and date('2001-03-29')
            and maturity between 75 and 105--= (select min(maturity) from spx where date(observation) = date('2000-09-18'))
            order by strike""",
            con)

def process(y0, dt, func, *args):
    """Feed me starting values, a differential equation like `lambda y_t, dt, dw[, ..more driving processes]`
    and a list of driving processes and I return your proces"""
    l = len(args[0])
    Y = np.ones(l+1)*y0
    for i, dargs in enumerate(zip(*args)):
        Y[i+1] = max(0.001, Y[i] + func(Y[i], dt, *dargs))
    return Y[1:]

class Simulation(object):
    def __init__(self, discretization_steps=100, T=1, n_samples=100, strikes_options=bla.loc[:,['strike','option']].values):
        self.discretization_steps = discretization_steps
        self.T = T
        self.n_samples = n_samples
        self.dt = T/discretization_steps
        self.W = np.random.normal(scale=np.sqrt(self.dt), size=(n_samples, discretization_steps, 3))
        self.strikes_options = strikes_options
        
    def evaluate(self, snu, delta, k, strike):
        """todo"""
        ########## estimated from market ##########
        qS = 0.008
        xi = 0.23
        ########## estimated from market ##########
        avg = 0
        for W in self.W:
            nu = process(
                xi,
                self.dt, 
                lambda nu, dt, dw: k*(xi-nu)*dt + np.sqrt(nu)*snu@dw,
                W
            )
            S = np.prod(1 + qS*self.dt +                           W.T[0] * np.sqrt(nu))
            M = np.prod(1 +            + (-qS*W.T[0] + delta*(snu @ W.T)) / np.sqrt(nu))
            avg += max(0, S*M - strike)
        return avg / self.n_samples
    
    def get_fit(self, params):
        snu = params[:3]
        delta = params[3]
        k = params[4]
        error = 0
        for strike, option in self.strikes_options:
            error += (self.evaluate(snu, delta, k, strike)-option)**2
        return error
        

a = Simulation()
a.get_fit(np.array([.2,.2,.2,.2,2.]))
from time import time
t = time()
m = minimize(a.get_fit, np.array([.2,.2,.2,.2,2.]))
print(time()-t)

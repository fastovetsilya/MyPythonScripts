#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:25:07 2019

@author: ilia
"""
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np

# Sample function to optimize
xs = np.linspace(-2, 10, 10000)

def f(x):
    noise = np.random.normal(0, 0.25, size=np.shape(x))
    result = np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + \
    1 / (x ** 2 + 1) + noise
    return result

#plt.scatter(xs, f(xs), s = 0.1)


# Adjust optimizer

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
#utility = UtilityFunction(kind="ei", kappa=0, xi=0.1)
#utility = UtilityFunction(kind="poi", kappa=0, xi=1e-2)

pbounds = {'x': (-2, 10)}

optimizer = BayesianOptimization(
    f=None,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

optimizer.set_gp_params(alpha = 10e-3, n_restarts_optimizer=0)
xlist = []
targetlist = []
errorlist = []

for i in range(30):
    print('Iteration: ', i)
    
    if i <= 10:
        print('Trying random value...')
        next_point_to_probe = {'x': np.random.randint(-2, 10) + np.random.rand()}
    else:
        next_point_to_probe = optimizer.suggest(utility)
        
    print("Next point to probe is:", next_point_to_probe)
    xlist.append(next_point_to_probe['x'])    
    
    predicted, sigma_pred = optimizer._gp.predict(
            np.array([next_point_to_probe['x']]).reshape(-1, 1), 
                                        return_std=True)
    print('Predict target to be: ', predicted, ' with std: ', sigma_pred)
    
    target = f(**next_point_to_probe)
    print("Found the target value to be:", target)
    targetlist.append(target)
    
    error = (predicted - target) ** 2
    errorlist.append(error)
    
    optimizer.register(
        params=next_point_to_probe,
        target=target,
    )
    
plt.plot(targetlist)
plt.plot(errorlist)

predicted, sigma_pred = optimizer._gp.predict(xs.reshape(-1, 1), return_std=True)

plt.plot(xs, predicted, 'g-')
plt.plot(xs, predicted + sigma_pred, 'r--')
plt.plot(xs, predicted - sigma_pred, 'r--')
plt.scatter(xs, f(xs), s = 0.5)
plt.scatter(xlist, targetlist, color='black', s = 5)











# Copyright 2022, Yi Han, All rights reserved.


import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, space_eval, hp, rand, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

space_params = {   
    'min_samples_split': hp.qloguniform('min_samples_split', 2, 8, 1),
    'max_depth':  hp.quniform('max_depth', 3, 30, 1) 
}


def ngbBayesVal(X_train, Y_train, X_dev, Y_dev, space_params = space_params, eta= 0.01,  max_eval = 10, seed = 1):
    
    
    def objective(params):
        params = {   
            'min_samples_split': int(params['min_samples_split']),
            'max_depth': int(params['max_depth'])
        }
        
        
        default_tree_learner2 = DecisionTreeRegressor(criterion='friedman_mse', # mod
                                                      min_samples_split =params['min_samples_split'],
                                                      max_depth=params['max_depth'])

        ngb_reg = NGBRegressor(learning_rate=eta , n_estimators= 400,
                               Dist=Normal, Score=MLE,
                               Base = default_tree_learner2)
        
        
        ngb = ngb_reg.fit(X = X_train, Y = Y_train, X_val = X_dev, Y_val = Y_dev, early_stopping_rounds=10)
        #Y_preds = ngb.predict(X_dev)
        Y_dists = ngb.pred_dist(X_dev)
        dev_NLL = -Y_dists.logpdf(Y_dev.flatten()).mean()
        
        print(f'Pars = {params}  \t NLL = {dev_NLL}')
        
        return {
            'loss': dev_NLL,
            'status': STATUS_OK
        }
    
    
    trials = Trials()
    best = fmin(fn=objective, # function of space, get pars using space['par']
            space=space_params,
            algo=tpe.suggest,
            trials = trials,
            rstate=np.random.RandomState(seed),
            max_evals=max_eval)
    
    
    best_params = space_eval(space_params, best) # this returns dictionary parameter 
    #best_params = trials.argmin # this returns space parameter 
    return best_params


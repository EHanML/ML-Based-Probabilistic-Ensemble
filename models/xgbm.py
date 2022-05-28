# Copyright 2022, Yi Han, All rights reserved.


import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, space_eval, hp, rand, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample
import xgboost as xgb
from sklearn.metrics import mean_squared_error

space_params = {   
    'eta': hp.choice('eta', [0.01,0.05]),
    'max_depth': hp.quniform('max_depth', 2, 12, 1),
    'gamma': hp.uniform('gamma', 0, .7),
    'min_child_weight': hp.loguniform('min_child_weight', 0, 5),
    'subsample': hp.choice('subsample', [0.6,0.8,1]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.6,0.8,1]),
}


def XGB_run(X_train, Y_train, X_dev, Y_dev, params,  num_boost_round = 1000, early_stopping_rounds = 100,  verbose = 100):
    
    dtrain = xgb.DMatrix(X_train, Y_train)
    ddev = xgb.DMatrix(X_dev, Y_dev)
    watchlist = [(dtrain, 'train'), (ddev, 'eval')]
    
    params = {   
            'booster': 'gbtree',
            'nthread': 10,
            'eta':  "{:.3f}".format(params['eta']),
            'tree_method': 'auto',
            'max_depth': int(params['max_depth']),
            'gamma': "{:.3f}".format(params['gamma']),
            'min_child_weight': "{:.3f}".format(params['min_child_weight']),
            'subsample': "{:.3f}".format(params['subsample']),
            'colsample_bytree': "{:.3f}".format(params['colsample_bytree'])
            }    
    model = xgb.train(params = params,
                      dtrain = dtrain,
                      num_boost_round =num_boost_round,
                      evals = watchlist,
                      maximize=False,
                      early_stopping_rounds = early_stopping_rounds,
                      verbose_eval=verbose)
    return model, model.best_ntree_limit, model.best_score


def xgbBayesVal(X_train, Y_train, X_dev, Y_dev, space_params = space_params, max_eval = 10, seed = 1):
    
    def objective(params):

        xgbfit, ntree, score  = XGB_run(X_train, Y_train, X_dev, Y_dev, params, verbose = False)
        print(f'Pars = {params} \t ntree = {ntree} \t RMSE = {score}')
        return {
            'loss': score,
            'status': STATUS_OK,
            # other results
            'ntree': ntree
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
    return best_params, trials.best_trial.get('result').get('ntree')


def XGB_run2(X_train, Y_train, X_dev, Y_dev , mu_model, params, num_boost_round = 1000, early_stopping_rounds = 100,  verbose = 100):
    
    dtrain = xgb.DMatrix(X_train, Y_train)
    ddev = xgb.DMatrix(X_dev, Y_dev)
    watchlist = [(dtrain, 'train'), (ddev, 'eval')]
    
    params = {   
            'booster': 'gbtree',
            'nthread': 10,
            'eta':  "{:.3f}".format(params['eta']),
            'tree_method': 'auto',
            'max_depth': int(params['max_depth']),
            'gamma': "{:.3f}".format(params['gamma']),
            'min_child_weight': "{:.3f}".format(params['min_child_weight']),
            'subsample': "{:.3f}".format(params['subsample']),
            'colsample_bytree': "{:.3f}".format(params['colsample_bytree'])
        }
    
    def obj_logtheta(preds, train_data):
        y = train_data.get_label()
        miu =  mu_model.predict(train_data)
        grad = 1 - np.exp(-2*preds) * ((y - miu)**2)
        hess =  2 * np.exp(-2*preds) * ((y - miu)**2)
        return grad, hess

    def eval_NNL(preds, train_data):
        y = train_data.get_label()
        miu =  mu_model.predict(train_data)
        pi =  np.full(fill_value= np.pi, shape=np.shape(y)[0],dtype=float)
        nll = ( np.sum(0.5 *np.log(2*pi) + preds + 0.5 * ((y - miu)**2) * np.exp(-2*preds)))/y.shape[0]
        return 'NLL', nll
    
    model = xgb.train(params = params,
                      dtrain = dtrain,
                      num_boost_round =  num_boost_round,
                      evals = watchlist,
                      maximize=False,
                      early_stopping_rounds = early_stopping_rounds,
                      obj= obj_logtheta,
                      feval= eval_NNL,
                      verbose_eval=verbose)
    
    return model, model.best_ntree_limit, model.best_score  

def xgbBayesVal2(X_train, Y_train, X_dev, Y_dev, mu_model, space_params = space_params, max_eval = 10, seed = 1):

    def objective(params):
        
        xgbfit, ntree, score  = XGB_run2(X_train, Y_train, X_dev, Y_dev, mu_model, params, verbose = False)
        print(f'Pars = {params} \t ntree = {ntree} \t NLL = {score}')
        return {
            'loss': score,
            'status': STATUS_OK,
            # other results
            'ntree': ntree
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
    return best_params, trials.best_trial.get('result').get('ntree')
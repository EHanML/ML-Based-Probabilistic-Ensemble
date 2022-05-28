# Copyright 2022, Yi Han, All rights reserved.


import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, space_eval, hp, rand, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

space_params = {   
    'num_leaves': hp.qloguniform('num_leaves', 3, 11, 1),
    'max_depth': hp.choice('max_depth', [-1, 6, 10, 20, 30, 50]),
    'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 2, 8, 1),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1)   
}


def LGB_run(X_train, Y_train, X_dev, Y_dev, params, num_boost_round = 1000, early_stopping_rounds = 100, predictors = None, verbose = 100, seed = 1):
    
    train = lgb.Dataset(data=X_train,
                        label=Y_train,
                        feature_name = predictors,
                        # feature_name='auto'
                        group=None, categorical_feature=['E_Label_ParcelID'])
    
    dev = lgb.Dataset(data=X_dev,
                        label=Y_dev,
                        reference=train,
                        feature_name = predictors,
                      # feature_name='auto',
                        group=None, 
                        categorical_feature=['E_Label_ParcelID'])
    
    params = {   
    'objective': 'regression',
    #'reg_sqrt': True, 
    'boosting': 'gbdt',
    'boost_from_average': True,
    'metric':'rmse',
    'tree_learner': 'serial',
    'nthread': 8, # for the best speed, set this to the number of real CPU cores, not the number of thread
    'device_type': 'cpu',
    'seed': seed,
    'eta': 0.03,   
    #'max_bin': 500,
    'num_leaves': int(params['num_leaves']),
    'max_depth': int(params['max_depth']),
    'min_data_in_leaf': int(params['min_data_in_leaf']),
    'bagging_fraction': "{:.3f}".format(params['bagging_fraction']),
    'feature_fraction': "{:.3f}".format(params['feature_fraction'])
    }
    
    #sklearn
    model = lgb.train(params = params,
                      train_set = train,
                      num_boost_round = num_boost_round,
                      valid_sets = [dev],
                      valid_names = ['eval'],
                      early_stopping_rounds = early_stopping_rounds,
                      verbose_eval = verbose)
    
    score = model.best_score['eval']['rmse']
    
    ntree =  model.best_iteration
    
    return model, ntree, score


def lgbBayesVal(X_train, Y_train, X_dev, Y_dev, space_params= space_params, predictors =None, max_eval = 10, seed = 1):
    
    def objective(params):

        
        lgbfit, ntree, score  = LGB_run(X_train, Y_train, X_dev, Y_dev, params, predictors =predictors, verbose = False)
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



def LGB_run2(X_train, Y_train, X_dev, Y_dev, mu_model, params, num_boost_round = 1000, early_stopping_rounds = 10, predictors = None, verbose = 100, seed = 1):
    
    def obj_logtheta(preds, train_data):
        y = train_data.get_label()
        miu =  mu_model.predict(train_data.get_data())
        grad = 1 - np.exp(-2*preds) * ((y - miu)**2)
        hess =  2 * np.exp(-2*preds) * ((y - miu)**2)
        return grad, hess

    def eval_NNL(preds, train_data):
        y = train_data.get_label()
        miu =  mu_model.predict(train_data.get_data())
        pi =  np.full(fill_value= np.pi, shape=np.shape(y)[0],dtype=float)
        nll = ( np.sum(0.5 *np.log(2*pi) + preds + 0.5 * ((y - miu)**2) * np.exp(-2*preds)))/y.shape[0]
        return 'NLL', nll, False
    
    train = lgb.Dataset(data=X_train,
                        label=Y_train,
                        feature_name = predictors,
                        free_raw_data=False,
                        # feature_name='auto'
                        group=None, categorical_feature=['E_Label_ParcelID'])
    
    dev = lgb.Dataset(data=X_dev,
                            label=Y_dev,
                            reference=train,
                            feature_name = predictors,
                            free_raw_data=False,
                          # feature_name='auto',
                            group=None, 
                            categorical_feature=['E_Label_ParcelID'])
    
    params = {   
            'boosting': 'gbdt',
            'tree_learner': 'serial',
            'num_thread': 10,  
            'device_type': 'cpu',
            'seed': seed,
            'eta': 0.1,   
            #'max_bin': 500,
            'num_leaves': int(params['num_leaves']),
            'max_depth': int(params['max_depth']),
            'min_data_in_leaf': int(params['min_data_in_leaf']),
            'bagging_fraction': "{:.3f}".format(params['bagging_fraction']),
            'feature_fraction': "{:.3f}".format(params['feature_fraction'])
    }
    
    #sklearn
    model = lgb.train(params = params,
                      train_set = train,
                      num_boost_round = num_boost_round,
                      valid_sets = [dev],
                      valid_names = ['eval'],
                      fobj= obj_logtheta,
                      feval= eval_NNL,
                      early_stopping_rounds = early_stopping_rounds,
                      verbose_eval = verbose)
    
    score = model.best_score['eval']['NLL']
    
    ntree =  model.best_iteration
    
    return model, ntree, score

def lgbBayesVal2(X_train, Y_train, X_dev, Y_dev, mu_model,  space_params= space_params, predictors =None, max_eval = 10, seed = 1):
    
    def objective(params):
        
        
        lgbfit, ntree, score  = LGB_run2(X_train, Y_train, X_dev, Y_dev, mu_model,  params, predictors =predictors, verbose = False)
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
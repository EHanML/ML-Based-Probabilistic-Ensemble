# Copyright 2022, Yi Han, All rights reserved.


import warnings
warnings.filterwarnings("ignore")

import math
from scipy.special import logsumexp
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from hyperopt import fmin, tpe, space_eval, hp, rand, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample
from sklearn.metrics import mean_squared_error

import time


class ancRF:

    def __init__(self, X_train, y_train,  n_trees, hy_pars, tau = None, seed = 1, normalize = False):

        """
            Constructor for the class implementing a Bayesian RF..

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_trees      Number of trees to grow.
            @param hy_pars      [colsample, std_prior, lambdal2] 
            @param tau          Model precision. 
                                1) None if need to calibrated by val. 
                                2) Provide value (estimated by val) during the test time.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.

        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
            self.mean_y_train = np.mean(y_train)
            self.std_y_train = np.std(y_train)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ]) # sd = 1
            self.mean_X_train = np.zeros(X_train.shape[ 1 ]) # mu = 0
            self.mean_y_train = np.zeros(1)
            self.std_y_train = np.ones(1)

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin = 2).T
        
        # We construct the model
        self.colsample = hy_pars['colsample']
        self.lambdal2 = hy_pars['lambdal2']
        self.tau = tau
        
        if 'std_prior' in hy_pars:
            self.std_prior = hy_pars['std_prior']
        elif 'std_alpha' in hy_pars:
            self.std_alpha = hy_pars['std_alpha']
            if self.lambdal2 == 0:
                self.std_prior = 0
            else:
                self.std_prior = self.std_alpha / self.lambdal2
        else:
            print('Error: need either std_prior or std_alpha provided in the hyper-parameter space.')
        
        theta_ancs = norm.rvs(0,self.std_prior,size = n_trees)
        
        # Fit
        start_time = time.time()
        
        model = []
        
        for ti in range(n_trees):
            
            # pars
            theta_anc = theta_ancs[ti]
        
            params = {            
                'objective': 'reg:squarederror',
                'booster': 'gbtree',
                'tree_method': 'auto',
                'nthread': 10,
                'learning_rate': 1,
                'num_parallel_tree': 1,
        
                # ensure fully grow
                'max_depth': 1000, # i.e. unlimite < 2**1000 obs
                'min_child_weight': 0,
                'gamma':0,
            
#             'grow_policy': 'lossguide',
#             'subsample': 0.8,

                # hyperpars
                'colsample_bynode': self.colsample,
                'lambda': self.lambdal2,
                'alpha': self.lambdal2 * theta_anc  
            }
            
            # data
            # bagging
            np.random.seed(ti)
            sample_indice = np.random.choice(X_train.shape[0], X_train.shape[0], replace=True)
            X_train_bag = X_train[sample_indice,:]
            y_train_bag = y_train[sample_indice]
            dtrain = xgb.DMatrix(X_train_bag, y_train_bag)
            
            # we grow trees
            model_ti = xgb.train(params = params,
                                 dtrain = dtrain,
                                 num_boost_round = 1,
                                 maximize=False)
            
            model.append(model_ti)
            
            # [end] for ti in range(n_trees)
            
        
        # add attributes
        self.model = model
        self.n_trees = n_trees
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian RF.

            @param X_test   The matrix of features for the test data
            @param y_test   Labels for the test data. 

        """

        X_test = np.array(X_test, ndmin = 2)
#         y_test = np.array(y_test, ndmin = 2).T # for NN
        y_test = np.array(y_test) # for skl
    
        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)
        
        dtest = xgb.DMatrix(X_test, y_test)


        model = self.model
        
        y_hat_MC = np.stack([mdl.predict(dtest) for mdl in model]) # shape: (ntree,n_test)
        y_hat_MC = y_hat_MC * self.std_y_train + self.mean_y_train # reverse scale
        
        y_hat = y_hat_MC.mean(axis=0)
        y_true = y_test.squeeze()
        
        T = y_hat_MC.shape[0] # self.n_trees
        
        if self.tau == None:
            # tune tau
            grid_ll = []
            taul, tl = [], []
            for t in range(100):
                tau = 2**(-t)
                ll = (logsumexp(-0.5 * tau * (y_true - y_hat_MC)**2., 0) - np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
                test_ll = np.mean(ll)
                taul.append(tau)
                grid_ll.append(test_ll)
                tl.append(t)
            best_t = tl[grid_ll.index(max(grid_ll))]
            best_tau = taul[grid_ll.index(max(grid_ll))]
            G = 100
            gg = (taul[grid_ll.index(max(grid_ll))-1] - taul[grid_ll.index(max(grid_ll))+1])/G
            grid_ll = []
            taul, tl = [], []
            for t in range(G):
                tau = 2**(-best_t-1) +  gg*t
                ll = (logsumexp(-0.5 * tau * (y_true - y_hat_MC)**2., 0) - np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
                test_ll = np.mean(ll)
                taul.append(tau)
                grid_ll.append(test_ll)
                tl.append(t)
    
            LL = max(grid_ll)
            best_tau = taul[grid_ll.index(max(grid_ll))]
        else:
            tau = self.tau
            best_tau = self.tau
            ll = (logsumexp(-0.5 * tau * (y_true - y_hat_MC)**2., 0) - np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
            LL = np.mean(ll)
            
        rmse = np.sqrt(mean_squared_error(y_true,y_hat))

        # We are done!
        return  best_tau, rmse, -LL

def BayesVal(X_train, y_train, X_validation, y_validation, space_params, n_trees, FUN, max_eval = 10, seed = 1):
    
    """
    Return A, B, C
    
    A:    Best Hyperpars
    B:    RMSE
    C:    Estimated tau
    """
    
    def objective(params):
        
        # We fit the model
        model = FUN(X_train, y_train, n_trees = n_trees, hy_pars = params, tau = None) # modify
        
        # We obtain the performances from the validation sets
        tau, errors, nll = model.predict(X_validation, y_validation)
        
        print(f'Pars: {params} \t tau: {np.round(tau,2)} \n rmse: {np.round(errors,2)} \t NLL: {np.round(nll,2)}')
        
        return {
            'loss': nll,
            'status': STATUS_OK,
            # other results
            'tau': tau,
            'rmse': errors
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
    
    return best_params, trials.best_trial.get('result').get('rmse'), trials.best_trial.get('result').get('tau')
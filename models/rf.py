# Copyright 2022, Yi Han, All rights reserved.

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.special import logsumexp
from scipy.stats import norm
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import time


class rf:

    def __init__(self, X_train, y_train,  mtry, tau = None, n_trees = 1000, seed = 1, normalize = False):

        """
            Constructor for the class implementing a Bayesian RF..

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_trees      Number of trees to grow.
            @param mtry         Number of features considered at each node
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
        np.random.seed(seed=seed)
        params = {'max_features': int(mtry)} 
        model = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=10, max_depth=None, **params)

        # Fit
        start_time = time.time()
        model.fit(X_train, y_train_normalized)
        
        # add attributes
        self.model = model
        self.running_time = time.time() - start_time
        self.tau = tau
        self.seed = seed

        # We are done!

    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian RF.

            @param X_test   The matrix of features for the test data
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin = 2)
        y_test = np.array(y_test) # for skl

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        model = self.model
        
        standard_pred = model.predict(X_test)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train # reverse scale
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5

        y_probas =np.stack([reg_i.predict(X_test) for reg_i in model.estimators_])
        # prepare vars for tune sig
        T = y_probas.shape[0]
        
        y_hat_MC= y_probas
        y_hat = standard_pred
        y_true = y_test.squeeze()
        
        if self.tau == None:
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
        return  best_tau, rmse, -LL, y_hat_MC
    
    def predict_MC(self, X_test, y_test, nsample = 10, alpha= 0.05):
        np.random.seed(seed=self.seed)
        tau, errors, nll, y_hat_MC = self.predict(X_test, y_test)
        sig = np.sqrt(1/tau)
        y_MC_mixture = np.concatenate([np.random.normal(loc=samplei,scale=sig,size=(nsample,len(samplei))) for samplei in y_hat_MC] , axis=0)
        y_hat = y_hat_MC.mean(axis=0)
        sig_hat = y_MC_mixture.std(axis=0)
        L_hat = np.quantile(y_MC_mixture,alpha/2,axis=0)
        U_hat = np.quantile(y_MC_mixture,(1-alpha/2),axis=0)
        py_all = np.stack([norm.pdf(y_test, loc=samplei, scale=sig) for samplei in y_hat_MC])
        p_y = py_all.mean(axis=0)

        return y_hat, L_hat, U_hat, p_y, sig_hat, y_MC_mixture
    
    
    
def gridRF(X_train, y_train, X_dev, y_dev, n_trees=1000, max_eval = 10, seed = 1):
    np.random.seed(seed=seed)
    
    nump = X_train.shape[1]
    max_eval = max_eval
    mtrys = [i+1 for i in range(nump)]
    if nump > max_eval:
        step = nump//max_eval
        mtrys = mtrys[::step]
    
    best_nll = float('inf') 
    for mtry in mtrys:

        model = rf(X_train, y_train, n_trees = n_trees, mtry = mtry, tau = None) 
        tau, errors, nll, _ = model.predict(X_dev, y_dev)
        print ('mtry: {} \t NLL: {} \t tau:{} \t erorr: {}'.format(mtry,nll,tau,errors) )
        
        if (nll < best_nll): # val based on ll rather than RMSE
            best_nll = nll
            best_model = model
            best_tau = tau
            best_mtry = mtry
    return best_model, best_tau, best_mtry, best_nll
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
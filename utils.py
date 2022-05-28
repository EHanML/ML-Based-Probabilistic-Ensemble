#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Utility functions.
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import feather
import pandas as pd

def bool2int(bool):
    boolint = np.full(fill_value=1,shape=len(bool))
    boolint[~bool] =0
    return boolint
    
def get_NOIS(y_true,y_pred,y_l,y_u,alpha):
    awpi = np.mean(y_u - y_l)
    nois = awpi+ (2/alpha)*np.mean(np.multiply((y_l- y_true), bool2int(y_l > y_true)) + 
                          np.multiply((y_true - y_u), bool2int(y_true > y_u))  )
    return nois


def renew_MF(path):
    mf = feather.read_dataframe(path)
    nmf = mf.assign(p_y = norm.pdf(mf['Y'], loc=mf['y_hat'],scale=mf['sig_hat']))
    return nmf

def outframe(set_name,y,y_hat,sig_hat):
    WU2 = feather.read_dataframe('Data/WU.feather')
    all_test_MF = pd.DataFrame({
        'Index': WU2[WU2.loc[:,'Set']==set_name].loc[:,'Index'].values,
        'Y': y,
        'y_hat': y_hat,
        'sig_hat': sig_hat,
        'p_y': norm.pdf(y,loc=y_hat,scale=sig_hat)
    })
    return all_test_MF

def get_RMSE_NLL_NOIS_AWPI_ECPI(y_true,y_pred,y_l,y_u,alpha=0.05):
    awpi = np.mean(y_u - y_l)
    nois = awpi+ (2/alpha)*np.mean(np.multiply((y_l- y_true), bool2int(y_l > y_true)) + 
                          np.multiply((y_true - y_u), bool2int(y_true > y_u))  )
    ecpi = np.sum((y_l <= y_true) & (y_true <= y_u) )/len(y_true)
    
    sd = (y_u - y_pred )/norm.ppf(1- alpha/2)
    py = norm.pdf(y_true,loc = y_pred, scale=sd)
    nll = - np.log(py).mean()
    print(" & " + " & ".join(str(i) for i in np.round([np.sqrt(mean_squared_error(y_true,y_pred)), nll, nois, awpi, ecpi],2)) + " & 95\\% \\\\")
    return  np.sqrt(mean_squared_error(y_true,y_pred)), nll, nois, awpi, ecpi


def add_group_lags(data,
                   group, # a feature string or a list of feature string
                   timeindex, # a string indicating time index
                   features, # a list of string for generating lags
                   other, # a list of other feature strings or None
                   nlag, # number of lags
                   fillna = None
                  ):
    

    for i, d in enumerate(data.groupby(group)):
        n,g=d
        maxt = g.loc[:,timeindex].max()
        mint = g.loc[:,timeindex].min()
        #maxl = maxt-mint+1
        npd = pd.DataFrame({
            group: n,
            timeindex:(np.arange(mint, maxt+1)).astype('int32')
        })
        if(i == 0):
            out = npd
        else:
            out = pd.concat([out,npd],axis=0)
    
    if other == None:
        selectp = [group,timeindex]+features
    else:
        selectp = [group,timeindex]+other+features
    dfull = pd.merge(out,data.loc[:,selectp],on=[group,timeindex],how='left')
    
    print('Time range extension done!')
    
    ngrp = data[group]
    
    bar = progressbar.ProgressBar(maxval=len(data[group].unique())+10,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    bar.start()  
    
    for i, d in enumerate(dfull.groupby(group)):
        
        n,g0=d
        g = g0
        fn = g0.loc[:,features].columns
        
        for lag in np.arange(1,nlag+1):
            
            glag = g0.loc[:,features].shift(lag)
            glag.columns = fn +'_lag%s'%(lag)
            g = pd.concat([g,glag], axis=1)
        
        
        if(i == 0):
            out2 = g
        else:
            out2 = pd.concat([out2,g],axis=0)
        # we update bar 
        bar.update(i+1) 
        sleep(0.1)
        
    bar.finish()
                  
    if fillna != None:
        out2 = out2.fillna(fillna)     
    
    return out2


def add_group_leadlags(data,
                   group, # a feature string or a list of feature string
                   timeindex, # a string indicating time index
                   lag_features, # a list of string for generating lags
                   lead_features, # a list of string for generating leads
                   other, # a list of other feature strings
                   nlag, # number of lags
                   nlead, # number of lags
                   fillna = None
                  ):
    # expand time range 
    # make sure shift with same resolution
    for i, d in enumerate(data.groupby(group)):
        n,g=d
        maxt = g.loc[:,timeindex].max()
        mint = g.loc[:,timeindex].min()
        #maxl = maxt-mint+1
        npd = pd.DataFrame({
            group: n,
            timeindex:(np.arange(mint, maxt+1)).astype('int32')
        })
        if(i == 0):
            out = npd
        else:
            out = pd.concat([out,npd],axis=0)
    
    # merge data with the expanded time frame
    merged_list = list(pd.Series(lag_features+lead_features).unique())
    dfull = pd.merge(out,data.loc[:,[group,timeindex]+other + merged_list],on=[group,timeindex],how='left')
     
   
    for i, d in enumerate(dfull.groupby(group)):
        n,g=d
        
        # add lags 
        for p in lag_features:
            for lag in np.arange(1,nlag+1):
                #rint('%s%s_lag%s'%(n,p,lag))
                pname = '%s_lag%s'%(p,lag)
                g.loc[:,pname] = g.loc[:,p].shift(lag)
        
        # add leads
        for p in lead_features:
            for lead in np.arange(0,nlead):
                #print('%s%s_lead%s'%(n,p,lag))
                pname = '%s_lead%s'%(p,lead)
                g.loc[:,pname] = g.loc[:,p].shift(-lead) 
        
        if(i == 0):
            out2 = g
        else:
            out2 = pd.concat([out2,g],axis=0)
            
    if fillna != None:
        out2 = out2.fillna(fillna)     
    
    return out2

def add_group_lags(data,
                   group, # a feature string or a list of feature string
                   timeindex, # a string indicating time index
                   features, # a list of string for generating lags
                   other, # a list of other feature strings
                   nlag, # number of lags
                   fillna = None
                  ):

    for i, d in enumerate(data.groupby(group)):
        n,g=d
        maxt = g.loc[:,timeindex].max()
        mint = g.loc[:,timeindex].min()
        #maxl = maxt-mint+1
        npd = pd.DataFrame({
            group: n,
            timeindex:(np.arange(mint, maxt+1)).astype('int32')
        })
        if(i == 0):
            out = npd
        else:
            out = pd.concat([out,npd],axis=0)
    
    dfull = pd.merge(out,data.loc[:,[group,timeindex]+other+features],on=[group,timeindex],how='left')
        
    for i, d in enumerate(dfull.groupby(group)):
        n,g=d
        for p in features:
            for lag in np.arange(1,nlag+1):
                #print('%s%s_lag%s'%(n,p,lag))
                pname = '%s_lag%s'%(p,lag)
                g.loc[:,pname] = g.loc[:,p].shift(lag)
        #print(n)
        #print(g)
        if(i == 0):
            out2 = g
        else:
            out2 = pd.concat([out2,g],axis=0)
            
    if fillna != None:
        out2 = out2.fillna(fillna)     
    
    return out2



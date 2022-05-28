#                     rm(list=ls()); sapply(dev.list(), dev.off); cat('\014')

library(tidyverse)
library(magrittr)
library(caret)
library(feather)
library(mgcv)
library(glmnet)
library(nnls)
library(RcppCNPy)


mypath ="C:/Users/Yi/Documents/Research/Ah_Research/WaterUsage/"
setwd(mypath)
source("mysource.R")

# load samples

lstm_mc <- npyLoad("Py3_sim/Out_test_MF/month12/LSTM_MC.npy")
dim(lstm_mc)

mlp_mc <- npyLoad("Py3_sim/Out_test_MF/month12/MLP_MC.npy")
dim(lstm_mc)

rf_mc <- npyLoad("Py3_sim/Out_test_MF/month12/RF_MC.npy")
dim(lstm_mc)


setwd(paste(mypath, "Py3_sim/Ensemble/", sep ='')) # modify
load('metalist.rda')

t = metalist$m12_dev_yhat
getRMSE(t$Y,t$NGB)

# train 12 month

###### glmnet ######
X_train = metalist$m12_dev_yhat %>% select(-Index,-Y)
Y_train = metalist$m12_dev_yhat %>% select(Y)
X_test =  metalist$m12_test_yhat %>% select(-Index,-Y) %>% select(-contains('ing'))
Y_test =  metalist$m12_test_yhat %>% select(Y)
p= dim(X_train)[2]


predictors <- names(X_train)
predictors
predictors <- names(X_test)
predictors


get_stack_sample <- function(w){
  
  fyhat = metalist$m12_test_yhat %>% select(-Index, -Y,-contains('ing'))
  fsighat = metalist$m12_test_sig %>% select(-Index, -Y)
  fall = bind_cols(fyhat,fsighat)
  stack_sample = metalist$m12_test_yhat %>% select(Index) %>% as.matrix()
  
  power = 4
  wi = round(w,power)
  n  = 10^power
  wn = wi*n
  sum(wn)
  
  
  for(m in 1:p){
    
    nm = wn[m] # model weights 
    name = names(fall)[m]
    
    if(grepl('LSTM_MC',name) ){
      
      ml_sample <- t(apply(lstm_mc,2, function(v) sample(v,size = nm) )) 
      
    }else if(grepl('MLP_MC',name)){
      
      ml_sample <- t(apply(mlp_mc,2, function(v) sample(v,size = nm) ))
      
    }else if(grepl('RF',name)){
      
      ml_sample <- t(apply(rf_mc,2, function(v) sample(v,size = nm) ))
      
    }else{
      
      mu_sig = fall[c(m,m+p)]
      ml_sample = t(apply(mu_sig, 1, function(v) rnorm(n=nm,mean = v[1], sd = v[2])  ))
      
    }
    
    if(nm==0){
      
      stack_sample = stack_sample
      
    }else{
      
      stack_sample = cbind(stack_sample,ml_sample)
      
    }
    print(name) #  stack_sample %>% dim
    
  }
  
  return(stack_sample)
}


get_LU <- function(df, alpha = 0.05){
  
  get_LUv <- function(v) quantile(v,c(alpha/2, 1-alpha/2))
  df_LU <- t(apply(df,1,get_LUv))
 # df_LU[df_LU <0] = 0
  
  return(df_LU)
}



get_results <- function(L,U,y,alpha=0.05){
  
  
  awpi = mean(U -L)
  ecpi = mean(   (L <= y) & (U >= y)     )
  
  nois = awpi + (2/alpha) * ( sum((L - y)[L > y])  +  sum((y - U)[y > U])  )/length(y)
  
  return(c(nois,awpi,ecpi))
  
}




###### NNLS ######
library(nnls)

fit.nnls <- nnls::nnls(A= (X_train %>% as.matrix()),b=(Y_train %>% as.matrix()))
init.coef.nnls <- stats::coef(fit.nnls)
init.coef.nnls
nnlsprd <- (X_test  %>% as.matrix()) %*%   as.matrix(init.coef.nnls)

getRMSE(obs =Y_test %>% unlist(), prd = nnlsprd ) # 1850.975




#NLL
w= as.matrix(init.coef.nnls)
BMA_pr = metalist$m12_test_py %>% select(-Index,-Y,-contains('ing')) %>% as.matrix() %*% w
nnls_py <- BMA_pr
BMA_NLL = -mean(log(BMA_pr))
round(BMA_NLL,2) #  8.88


pic = ggplot()+
  geom_bar(mapping = aes(x=predictors, y=w) , width = 0.7,stat = "identity", fill = "dodgerblue3",color="dodgerblue3")+
  theme_light()+
  theme(axis.text.y = element_text(size =15, face = 'bold'))+
  labs(y="Weight", x= NULL, title = 'Stacking of Means (k-months)') +
  coord_flip()
pic

{
  setwd(paste(mypath,  "Py3_sim/Ensemble/plots/", sep =''))
  png(filename = sprintf("month12_nnls_stack_sim.png" ),
      width =400, height =600)
  
  print( pic)
  dev.off()
}


# LU
stack_sample = get_stack_sample(w %>% unlist)
stack_sample %>% dim()

# 90%
alpha = 0.1
LU_df <- get_LU(stack_sample,alpha )

get_results(LU_df[,1],LU_df[,2], metalist$m12_test_yhat$Y,alpha  ) #  8805.7740136 5156.6186404    0.8782974

# 95%
alpha = 0.05
LU_df <- get_LU(stack_sample,alpha )

get_results(LU_df[,1],LU_df[,2], metalist$m12_test_yhat$Y,alpha ) #  1.104734e+04 6.406516e+03 9.242035e-01

# to store
nnls_L   <- LU_df[,1]
nnls_U <- LU_df[,2]





###### avg ######

#Avgprd <- apply(X_test %>% select(-NGB,-contains('Probabilistic')),1, mean )
Avgprd <- apply(X_test,1, mean )
getRMSE(obs =Y_test %>% unlist(), prd =Avgprd  ) # 1883.941

w = matrix(rep(1/p,p))
BMA_pr = metalist$m12_test_py %>% select(-Index,-Y,-contains('ing')) %>% as.matrix() %*% w
avg_py = BMA_pr
BMA_NLL = -mean(log(BMA_pr))
round(BMA_NLL,2) # 8.8

# LU
stack_sample = get_stack_sample(w %>% unlist)
stack_sample %>% dim()

# 90%
alpha = 0.1
LU_df <- get_LU(stack_sample,alpha )

get_results(LU_df[,1],LU_df[,2], metalist$m12_test_yhat$Y,alpha  ) #  8538.1992349 6286.1526108    0.9162384

# 95%
alpha = 0.05
LU_df <- get_LU(stack_sample,alpha )

get_results(LU_df[,1],LU_df[,2], metalist$m12_test_yhat$Y,alpha ) #  10378.945786  7866.942207     0.955036


avg_L  <- LU_df[,1]
avg_U  <- LU_df[,2]




## pseudo-BMA ##
f1 <- log( metalist$m12_dev_py %>% select(-Index, -Y))
elpd <- apply(f1, 2, sum)
se_elpd <- apply(f1, 2, sd)
b =max(elpd )

w_pBMA = exp(elpd  - b)/sum(exp(elpd  - b))
w_pBMA

# ==> RF


mean(log(metalist$m12_test_py$AR1))








## stack BMA ##

X =  metalist$m12_dev_py %>% select(-Index, -Y) %>% as.matrix()
p = dim(X)[2]

softmax <- function(w) exp(w)/(sum(exp(w)))




obj <- function(w){
  
  
  b = matrix(softmax(w))
  
  L = -mean(log( X %*% b)) # + 0.999*abs( sum(b)-1 )
  
  return(L)
} 
set.seed(1)
# initopt <- (rnorm(p,0,0.0001))
initopt <- rep(1,p)
best_w =optim(initopt, obj, method ="BFGS" )$par
best_w =optim(initopt, obj, method ="L-BFGS-B" )$par
w = softmax(best_w)
best=obj(w)
best
best_w = w

# test
Bayes_stack_prd <- ( X_test %>% as.matrix()) %*% best_w
getRMSE(obs =Y_test %>% unlist(), prd = Bayes_stack_prd ) # 1908.048

w=best_w

BMA_pr = metalist$m12_test_py %>% select(-Index,-Y,-contains('ing')) %>% as.matrix() %*% w
dist_py = BMA_pr 
BMA_NLL = -mean(log(BMA_pr))
round(BMA_NLL,2)  # 8.74



pic = ggplot()+
  geom_bar(mapping = aes(x=predictors, y=w) , width = 0.7,stat = "identity", fill = "chocolate1",color="chocolate1")+
  theme_light()+
  theme(axis.text.y = element_text(size =15, face = 'bold'))+
  labs(y="Weight", x= NULL, title = 'Stacking of Distributions (k-months)') +
  coord_flip()
pic

{
  setwd(paste(mypath,  "Py3_sim/Ensemble/plots/", sep =''))
  png(filename = sprintf("month12_dist_stack_sim.png" ),
      width =400, height =600)
  
  print( pic)
  dev.off()
}



# LU
stack_sample = get_stack_sample(w %>% unlist)
stack_sample %>% dim()

# 90%
alpha = 0.1
LU_df <- get_LU(stack_sample,alpha )

get_results(LU_df[,1],LU_df[,2], metalist$m12_test_yhat$Y,alpha  ) #   8579.559719 5818.358855    0.893371

# 95%
alpha = 0.05
LU_df <- get_LU(stack_sample,alpha )

get_results(LU_df[,1],LU_df[,2], metalist$m12_test_yhat$Y,alpha ) #  1.061738e+04 7.474786e+03 9.358513e-01

dist_L <- LU_df[,1]
dist_U <- LU_df[,2]





# update metalist




My <- metalist$m12_test_yhat %>% 
  mutate(Averaging = Avgprd, Mean_Stacking = nnlsprd[,1], Dist_Stacking = Bayes_stack_prd[,1] )

metalist$m12_test_yhat <- My



ML <- metalist$m12_test_L %>% 
  mutate(Averaging = avg_L, Mean_Stacking = nnls_L, Dist_Stacking = dist_L )

metalist$m12_test_L  <- ML


MU <- metalist$m12_test_U %>% 
  mutate(Averaging = avg_U, Mean_Stacking = nnls_U, Dist_Stacking = dist_U)

metalist$m12_test_U  <- MU


Mpy <- metalist$m12_test_py %>% 
  mutate(Averaging = avg_py  %>% as.vector() , Mean_Stacking = nnls_py  %>% as.vector(), Dist_Stacking = dist_py  %>% as.vector())

metalist$m12_test_py  <- Mpy

mypath ="C:/Users/Yi/Documents/Research/Ah_Research/WaterUsage/"
setwd(paste(mypath, "Py3_sim/Ensemble/", sep ='')) 
save(metalist,file='metalist.rda')

metalist



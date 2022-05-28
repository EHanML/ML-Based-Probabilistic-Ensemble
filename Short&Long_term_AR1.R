#        rm(list=ls()); sapply(dev.list(), dev.off); cat('\014')

#########################
###   Tune ML model   ###
#########################

# AR: Template AA: selected by imp w/o FE

#########################
ar_order = 1
withinloopFix = F


# local work
library(tidyverse)



library(magrittr)
library(caret)
library(arrow)


mypath ='C:\\Users\\Yi\\Documents\\Research\\Ah_Research\\WaterUsage\\ML-Based Probabilistic Ensemble'
setwd(paste(mypath,"\\Data",sep = ""))

WU =read_feather("WU.feather")



getRMSE <- function(obs, prd){
  sqrt(mean((prd - obs)^2)) 
}





# (A) FE

# take log here!!!!

WU <- WU %>%  select(TotalWaterUse, ParcelID, year, month, monthIndex,Index,ET_turf_month_mean) %>%
              mutate(Y = TotalWaterUse ) %>%
           #  mutate_at(vars(matches("TotalWaterUse")), ~(log(.)) ) %>%
            select(Index, Y, TotalWaterUse, everything()) %>%
            arrange(ParcelID, monthIndex) # important for TS

            

WU %>% glimpse()


num_houses <- length(unique(WU$ParcelID))

# OUT (A): WU

##########################################################################################

# (B) Val stage 

ALlTimePoint <- WU$monthIndex %>% unique()
split <- ALlTimePoint %>% max -12*2 
split2 <- ALlTimePoint %>% max -12
# delete 2 when testing
#split <- ALlTimePoint %>% max -12

# train
WUTrain <- WU %>% filter(monthIndex <= split)

# val
WUTest <- WU %>% filter(monthIndex > split & monthIndex <= split2) 
TestLength <- WUTest$monthIndex %>% unique() %>% length


#OUT  WUTest WUTrain

##########################################################################################
# Fit AR(1) [start]
# all train fit

prepare_trainy <- function(data){
  
  
  ntime <-  data$monthIndex %>% unique %>% length
  trainy <- tibble(TotalWaterUse = rep(NA, ntime))
  insertNA <- tibble(TotalWaterUse = rep(NA, ntime))
  
  for(i in 1:length(unique( data$ParcelID))){
    cust <- unique( data$ParcelID)[i]
    custyi <-   data %>% 
      filter( ParcelID == cust) %>%  # select( ParcelID) %>% unique()
      select(TotalWaterUse)
    
    trainy <- bind_rows(trainy, custyi,insertNA)
    
  }
  
  trainy <- trainy %>% unlist %>% as.vector
  return(trainy)
}

fitAR <- function(data,ARorder){
  

  ntime <-  data$monthIndex %>% unique %>% length
  trainy <- tibble(TotalWaterUse = rep(NA, ntime))
  insertNA <- tibble(TotalWaterUse = rep(NA, ntime))
  
  for(i in 1:length(unique( data$ParcelID))){
    cust <- unique( data$ParcelID)[i]
    custyi <-   data %>% 
      filter( ParcelID == cust) %>%  # select( ParcelID) %>% unique()
      select(TotalWaterUse)
    
    trainy <- bind_rows(trainy, custyi,insertNA)
    
  }
  
  trainy <- trainy %>% unlist %>% as.vector
  
  fit.ar1.all <- arima(trainy, order=c(ARorder,0,0), method="ML")
  
  return(fit.ar1.all)
}

fit.ar1.all <-fitAR(WUTrain, ar_order)
fixed.phi <- as.numeric(fit.ar1.all$coef[1])
fixed.intercept <- as.numeric(fit.ar1.all$coef[2])


##########################################################################################

# (F) Test feature


WUTest$monthIndex %>% unique() 
MetaF_Test <- WUTest %>% mutate( AR1_1m =NA , AR1_1m_se = NA,AR1_12m =NA, AR1_12m_se=NA )

parcelIDpool = WU$ParcelID %>%unique()
# train all
for(icust in 1: length(parcelIDpool) ){
  
  # fix cust for this loop
  parcel <- parcelIDpool[icust]
  
  monthPool <-  WUTest$monthIndex %>% unique() %>% sort
  

  
  
  # 1 month pred
  NAhead <- 1
  parcel12prd <- array()
  parcel12prd1 <- array()
  
  
  for(m in 1: length(monthPool) ){
    
    tsY <- WU %>% 
      filter(monthIndex < monthPool[m]) %>%
      filter(ParcelID == parcel ) %>%  # select( ParcelID) %>% unique()
      select(TotalWaterUse) %>% unlist %>% as.vector # %>% length
    
    # yeah~ lets fit
    fit.random0 <- arima(tsY, order=c(ar_order,0,0), fixed=c(fixed.phi, NA), transform.pars=FALSE, method="ML")
    
    parcel12prd[m] <- as.double(predict( fit.random0, n.ahead = NAhead)$pred)
    
    
    parcel12prd1[m] <- as.double(predict(fit.random0, n.ahead = NAhead)$se)
    
    
    
  }
  
  MetaF_Test[ MetaF_Test$ParcelID == parcel, "AR1_1m"]  <-  parcel12prd
  MetaF_Test[ MetaF_Test$ParcelID == parcel, "AR1_1m_se"]  <-  parcel12prd1
  
  # 1 month pred [end]
  
  # 12 month pred
  
  NAhead <- 12

    tsY <- WUTrain %>% filter(ParcelID == parcel ) %$% TotalWaterUse %>% unlist %>% as.vector

    
    # yeah~ lets fit
    fit.random0 <- arima(tsY, order=c(ar_order,0,0), fixed=c(fixed.phi, NA), transform.pars=FALSE, method="ML")
    
    parcel12prd <- as.double(predict( fit.random0, n.ahead = NAhead)$pred)
    
    
    parcel12prd1 <- as.double(predict(fit.random0, n.ahead = NAhead)$se)
    
    
    


MetaF_Test[ MetaF_Test$ParcelID == parcel, "AR1_12m"]  <-  parcel12prd
MetaF_Test[ MetaF_Test$ParcelID == parcel, "AR1_12m_se"]  <-  parcel12prd1
  
  
  
  
  message(sprintf('Parcel %g is done',icust))
  
  
}



# val results
getRMSE(MetaF_Test$Y,( MetaF_Test$AR1_1m)) # result
getRMSE(MetaF_Test$Y,( MetaF_Test$AR1_12m)) # result


# save dev



alpha = 0.05

m1_result <- MetaF_Test %>% select(-contains('12m'),-ET_turf_month_mean) %>% 
  transmute(Index=Index,
            Y=Y,
            y_hat = AR1_1m,
            sig_hat = AR1_1m_se,
            L = qnorm(p = alpha/2, mean = AR1_1m,sd =  AR1_1m_se ),
            U =  qnorm(p = 1-alpha/2, mean = AR1_1m,sd =  AR1_1m_se ),
            p_y = dnorm(Y,y_hat,sig_hat)

            )  



m12_result <- MetaF_Test %>% select(-contains('1m'),-ET_turf_month_mean) %>% 
  transmute(Index=Index,
            Y =Y,
            y_hat = AR1_12m,
            sig_hat = AR1_12m_se,
            L =  qnorm(p = alpha/2, mean = AR1_12m,sd =  AR1_12m_se ),
            U =  qnorm(p = 1-alpha/2, mean = AR1_12m,sd =  AR1_12m_se ),
            p_y = dnorm(Y,y_hat,sig_hat)

  )  



write_feather(m1_result, 
              sink =  paste(mypath,"/Out_dev_MF/month1/AR1_dev_1m_dist.feather",sep = "") )


write_feather(m12_result, 
              sink =paste(mypath,"/Out_dev_MF/month12/AR1_dev_12m_dist.feather",sep = "") ) 



 ############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################



# (B) Test stage 

ALlTimePoint <- WU$monthIndex %>% unique()
#split <- ALlTimePoint %>% max -12*2 
#split2 <- ALlTimePoint %>% max -12
# delete 2 when testing
 split <- ALlTimePoint %>% max -12

# train
WUTrain <- WU %>% filter(monthIndex <= split)

# val
WUTest <- WU %>% filter(monthIndex > split ) 
TestLength <- WUTest$monthIndex %>% unique() %>% length


#OUT  WUTest WUTrain

##########################################################################################
# Fit AR(1) [start]
# all train fit

prepare_trainy <- function(data){
  
  
  ntime <-  data$monthIndex %>% unique %>% length
  trainy <- tibble(TotalWaterUse = rep(NA, ntime))
  insertNA <- tibble(TotalWaterUse = rep(NA, ntime))
  
  for(i in 1:length(unique( data$ParcelID))){
    cust <- unique( data$ParcelID)[i]
    custyi <-   data %>% 
      filter( ParcelID == cust) %>%  # select( ParcelID) %>% unique()
      select(TotalWaterUse)
    
    trainy <- bind_rows(trainy, custyi,insertNA)
    
  }
  
  trainy <- trainy %>% unlist %>% as.vector
  return(trainy)
}

fitAR <- function(data,ARorder){
  
  
  ntime <-  data$monthIndex %>% unique %>% length
  trainy <- tibble(TotalWaterUse = rep(NA, ntime))
  insertNA <- tibble(TotalWaterUse = rep(NA, ntime))
  
  for(i in 1:length(unique( data$ParcelID))){
    cust <- unique( data$ParcelID)[i]
    custyi <-   data %>% 
      filter( ParcelID == cust) %>%  # select( ParcelID) %>% unique()
      select(TotalWaterUse)
    
    trainy <- bind_rows(trainy, custyi,insertNA)
    
  }
  
  trainy <- trainy %>% unlist %>% as.vector
  
  fit.ar1.all <- arima(trainy, order=c(ARorder,0,0), method="ML")
  
  return(fit.ar1.all)
}

fit.ar1.all <-fitAR(WUTrain, ar_order)
fixed.phi <- as.numeric(fit.ar1.all$coef[1])
fixed.intercept <- as.numeric(fit.ar1.all$coef[2])


##########################################################################################

# (F) Test feature


WUTest$monthIndex %>% unique() 
MetaF_Test <- WUTest %>% mutate( AR1_1m =NA , AR1_1m_se = NA,AR1_12m =NA, AR1_12m_se=NA )

parcelIDpool = WU$ParcelID %>%unique()
monthPool <-  WUTest$monthIndex %>% unique() %>% sort

# train all
for(icust in 1: length(parcelIDpool) ){
  
  # fix cust for this loop
  parcel <- parcelIDpool[icust]
  
  # 1 month pred
  NAhead <- 1
  parcel12prd <- array()
  parcel12prd1 <- array()
  
  
  for(m in 1: length(monthPool) ){
    
    tsY <- WU %>% 
      filter(monthIndex < monthPool[m]) %>%
      filter(ParcelID == parcel ) %>%  # select( ParcelID) %>% unique()
      select(TotalWaterUse) %>% unlist %>% as.vector # %>% length
    
    # yeah~ lets fit
    fit.random0 <- arima(tsY, order=c(ar_order,0,0), fixed=c(fixed.phi, NA), transform.pars=FALSE, method="ML")
    
    parcel12prd[m] <- as.double(predict( fit.random0, n.ahead = NAhead)$pred)
    
    
    parcel12prd1[m] <- as.double(predict(fit.random0, n.ahead = NAhead)$se)
    
    
    
  }
  
  MetaF_Test[ MetaF_Test$ParcelID == parcel, "AR1_1m"]  <-  parcel12prd
  MetaF_Test[ MetaF_Test$ParcelID == parcel, "AR1_1m_se"]  <-  parcel12prd1
  
  # 1 month pred [end]
  
  # 12 month pred
  
  NAhead <- 12
  
  tsY <- WUTrain %>% filter(ParcelID == parcel ) %$% TotalWaterUse %>% unlist %>% as.vector
  
  
  # yeah~ lets fit
  fit.random0 <- arima(tsY, order=c(ar_order,0,0), fixed=c(fixed.phi, NA), transform.pars=FALSE, method="ML")
  
  parcel12prd <- as.double(predict( fit.random0, n.ahead = NAhead)$pred)
  
  
  parcel12prd1 <- as.double(predict(fit.random0, n.ahead = NAhead)$se)
  
  
  
  
  
  MetaF_Test[ MetaF_Test$ParcelID == parcel, "AR1_12m"]  <-  parcel12prd
  MetaF_Test[ MetaF_Test$ParcelID == parcel, "AR1_12m_se"]  <-  parcel12prd1
  
  
  
  
  message(sprintf('Parcel %g is done',icust))
  
  
}


# val results
getRMSE(MetaF_Test$Y,( MetaF_Test$AR1_1m)) # result
getRMSE(MetaF_Test$Y,( MetaF_Test$AR1_12m)) # result


# save dev

alpha = 0.05

m1_result <- MetaF_Test %>% select(-contains('12m'),-ET_turf_month_mean) %>% 
  transmute(Index=Index,
            Y=Y,
            y_hat = AR1_1m,
            sig_hat = AR1_1m_se,
            L =  qnorm(p = alpha/2, mean = AR1_1m,sd =  AR1_1m_se ),
            U =  qnorm(p = 1-alpha/2, mean = AR1_1m,sd =  AR1_1m_se ), 
            p_y = dnorm(Y,y_hat,sig_hat)
            
  )  

m12_result <- MetaF_Test %>% select(-contains('1m'),-ET_turf_month_mean) %>% 
  transmute(Index=Index,
            Y =Y,
            y_hat = AR1_12m,
            sig_hat = AR1_12m_se,
            L = qnorm(p = alpha/2, mean = AR1_12m,sd =  AR1_12m_se ),
            U =  qnorm(p = 1-alpha/2, mean = AR1_12m,sd =  AR1_12m_se ),
            p_y = dnorm(Y,y_hat,sig_hat)
            
  )  







write_feather(m1_result, 
              sink =  paste(mypath,"/Out_test_MF/month1/AR1_1m_dist.feather",sep = "") )


write_feather(m12_result, 
              sink =paste(mypath,"/Out_test_MF/month12/AR1_12m_dist.feather",sep = "") ) 

# check results



check_results = function(result_frame){
  alpha = 0.05
  
  L = result_frame$L
  U = result_frame$U
  y = result_frame$Y
  yh = result_frame$y_hat
  ls = result_frame$sig_hat
  py = result_frame$p_y
  
  rmse = getRMSE(y,yh)
  awpi = mean(U - L)
  ecpi = sum(y<=U & y>= L)/length(y)
  nois = awpi + (2/alpha) * ( sum((L - y)[L > y])  +  sum((y - U)[y > U])  )/length(y)
  nll = -mean(log(py))
  return(c(rmse,nll,nois,awpi,ecpi))
  
}


check_results(m1_result)
check_results(m12_result)


alpha = 0.05


m1_result <- MetaF_Test %>% select(-contains('12m'),-ET_turf_month_mean) %>% 
  transmute(Index=Index,
            Y=Y,
            y_hat = AR1_1m,
            sig_hat = AR1_1m_se,
            L =  qnorm(p = alpha/2, mean = AR1_1m,sd =  AR1_1m_se ),
            U =  qnorm(p = 1-alpha/2, mean = AR1_1m,sd =  AR1_1m_se ), 
            p_y = dnorm(Y,y_hat,sig_hat)
            
  )  

m12_result <- MetaF_Test %>% select(-contains('1m'),-ET_turf_month_mean) %>% 
  transmute(Index=Index,
            Y =Y,
            y_hat = AR1_12m,
            sig_hat = AR1_12m_se,
            L = qnorm(p = alpha/2, mean = AR1_12m,sd =  AR1_12m_se ),
            U =  qnorm(p = 1-alpha/2, mean = AR1_12m,sd =  AR1_12m_se ),
            p_y = dnorm(Y,y_hat,sig_hat)
            
  )  



check_results = function(result_frame){
  alpha = 0.05
  
  L = result_frame$L
  U = result_frame$U
  y = result_frame$Y
  yh = result_frame$y_hat
  ls = result_frame$sig_hat
  py = result_frame$p_y
  
  rmse = getRMSE(y,yh)
  awpi = mean(U - L)
  ecpi = sum(y<=U & y>= L)/length(y)
  nois = awpi + (2/alpha) * ( sum((L - y)[L > y])  +  sum((y - U)[y > U])  )/length(y)
  nll = -mean(log(py))
  return(c(rmse,nll,nois,awpi,ecpi))
  
}


check_results(m1_result)
check_results(m12_result)


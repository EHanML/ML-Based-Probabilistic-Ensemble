#                     rm(list=ls()); sapply(dev.list(), dev.off); cat('\014')

# this script generate metalist that make use of feathers produced in python3



library(tidyverse)
library(magrittr)
library(caret)
library(arrow)

mypath ="C:/Users/Yi/Documents/Research/Ah_Research/WaterUsage/"
setwd(mypath)
source("mysource.R")

metalist <- list()

alpha = 0.05



# 1 month val MF
# out: three df: metaFrame, metaL, metaU

setwd(paste(mypath, "Py3_sim/Out_dev_MF/month1/", sep ='')) # modify
file_ls <- list.files(path= paste(mypath,  "Py3_sim/Out_dev_MF/month1/", sep =''), pattern=".feather$") # modify

for (i in 1:length(file_ls)){
  
  df = read_feather(file_ls[i])
  namehere = file_ls[i] %>% str_remove("_dev_1m_dist.feather")  # modify
  
  if(sum(names(df) %in% c('L',"U") )== 0){
    
    # no L U, then generate one use sig_hat
    
    MFf <- df %>% select(Index, Y, y_hat)   
    Lf <-df %>% mutate(L = qnorm(alpha/2, y_hat, sig_hat)) %>% select(Index, Y, L)
    Uf <-df %>% mutate(U = qnorm(1-alpha/2, y_hat, sig_hat)) %>% select(Index, Y, U)   
    Pf <- df %>% select(Index, Y, p_y) 
    Sf <- df %>% select(Index, Y, sig_hat) 
    
    names(MFf )[3]  <-  namehere 
    names(Lf )[3]  <-  namehere 
    names(Uf )[3]  <-  namehere 
    names(Pf )[3]  <-  namehere 
    names(Sf )[3]  <-  namehere 
    
  }else{
    
    # have L U
    
    MFf <- df %>% select(Index, Y, y_hat)   
    Lf <-df %>% select(Index, Y, L)   
    Uf <-df %>% select(Index, Y, U)   
    Pf <- df %>% select(Index, Y, p_y) 
    Sf <- df %>% select(Index, Y, sig_hat) 
    
    names(MFf )[3]  <-  namehere 
    names(Lf )[3]  <-  namehere 
    names(Uf )[3]  <-  namehere 
    names(Pf )[3]  <-  namehere 
    names(Sf )[3]  <-  namehere 
  }

  if(i == 1){
    metaFrame <- MFf
    metaL <-Lf
    metaU <-Uf
    metaP <-Pf
    metaS <- Sf
  }else{
    metaFrame  <- left_join( metaFrame , MFf %>% select(-Y), by="Index"  )
    metaL  <-  left_join( metaL , Lf %>% select(-Y), by="Index"  )
    metaU  <-  left_join( metaU , Uf %>% select(-Y), by="Index"  )
    metaP <- left_join( metaP , Pf %>% select(-Y), by="Index"  )
    metaS <- left_join( metaS , Sf %>% select(-Y), by="Index"  )
  }
  
}

# metaL[metaL<0] = 0

m1_dev_yhat <-  metaFrame  # modify
m1_dev_L <- metaL  # modify
m1_dev_U <- metaU   # modify
m1_dev_py <- metaP  # modify

metalist$m1_dev_yhat = m1_dev_yhat
metalist$m1_dev_L = m1_dev_L
metalist$m1_dev_U = m1_dev_U
metalist$m1_dev_py = m1_dev_py
metalist$m1_dev_sig = metaS

# 12 month val MF
# out: three df: metaFrame, metaL, metaU

setwd(paste(mypath, "Py3_sim/Out_dev_MF/month12/", sep ='')) # modify
file_ls <- list.files(path= paste(mypath,  "Py3_sim/Out_dev_MF/month12/", sep =''), pattern=".feather$") # modify

for (i in 1:length(file_ls)){
  
  df = read_feather(file_ls[i])
  namehere = file_ls[i] %>% str_remove("_dev_12m_dist.feather")  # modify
  
  if(sum(names(df) %in% c('L',"U") )== 0){
    
    # no L U, then generate one use sig_hat
    
    MFf <- df %>% select(Index, Y, y_hat)   
    Lf <-df %>% mutate(L = qnorm(alpha/2, y_hat, sig_hat)) %>% select(Index, Y, L)
    Uf <-df %>% mutate(U = qnorm(1-alpha/2, y_hat, sig_hat)) %>% select(Index, Y, U)   
    Pf <- df %>% select(Index, Y, p_y) 
    Sf <- df %>% select(Index, Y, sig_hat) 
    
    names(MFf )[3]  <-  namehere 
    names(Lf )[3]  <-  namehere 
    names(Uf )[3]  <-  namehere 
    names(Pf )[3]  <-  namehere 
    names(Sf )[3]  <-  namehere 
    
  }else{
    
    # have L U
    
    MFf <- df %>% select(Index, Y, y_hat)   
    Lf <-df %>% select(Index, Y, L)   
    Uf <-df %>% select(Index, Y, U)   
    Pf <- df %>% select(Index, Y, p_y) 
    Sf <- df %>% select(Index, Y, sig_hat) 
    
    names(MFf )[3]  <-  namehere 
    names(Lf )[3]  <-  namehere 
    names(Uf )[3]  <-  namehere 
    names(Pf )[3]  <-  namehere 
    names(Sf )[3]  <-  namehere 
  }
  
  if(i == 1){
    metaFrame <- MFf
    metaL <-Lf
    metaU <-Uf
    metaP <-Pf
    metaS <- Sf
  }else{
    metaFrame  <- left_join( metaFrame , MFf %>% select(-Y), by="Index"  )
    metaL  <-  left_join( metaL , Lf %>% select(-Y), by="Index"  )
    metaU  <-  left_join( metaU , Uf %>% select(-Y), by="Index"  )
    metaP <- left_join( metaP , Pf %>% select(-Y), by="Index"  )
    metaS <- left_join( metaS , Sf %>% select(-Y), by="Index"  )
  }
  
}

#metaL[metaL<0] = 0

m12_dev_yhat <-  metaFrame  # modify
m12_dev_L <- metaL  # modify
m12_dev_U <- metaU   # modify
m12_dev_py <- metaP   # modify

metalist$m12_dev_yhat = m12_dev_yhat
metalist$m12_dev_L = m12_dev_L
metalist$m12_dev_U = m12_dev_U
metalist$m12_dev_py = m12_dev_py
metalist$m12_dev_sig = metaS

# 1 month test MF
# out: three df: metaFrame, metaL, metaU

setwd(paste(mypath, "Py3_sim/Out_test_MF/month1/", sep ='')) # modify
file_ls <- list.files(path= paste(mypath,  "Py3_sim/Out_test_MF/month1/", sep =''), pattern=".feather$") # modify

for (i in 1:length(file_ls)){
  
  df = read_feather(file_ls[i])
  namehere = file_ls[i] %>% str_remove("_1m_dist.feather")  # modify
  
  if(sum(names(df) %in% c('L',"U") )== 0){
    
    # no L U, then generate one use sig_hat
    
    MFf <- df %>% select(Index, Y, y_hat)   
    Lf <-df %>% mutate(L = qnorm(alpha/2, y_hat, sig_hat)) %>% select(Index, Y, L)
    Uf <-df %>% mutate(U = qnorm(1-alpha/2, y_hat, sig_hat)) %>% select(Index, Y, U)   
    Pf <- df %>% select(Index, Y, p_y) 
    Sf <- df %>% select(Index, Y, sig_hat) 
    
    names(MFf )[3]  <-  namehere 
    names(Lf )[3]  <-  namehere 
    names(Uf )[3]  <-  namehere 
    names(Pf )[3]  <-  namehere 
    names(Sf )[3]  <-  namehere 
    
  }else{
    
    # have L U
    
    MFf <- df %>% select(Index, Y, y_hat)   
    Lf <-df %>% select(Index, Y, L)   
    Uf <-df %>% select(Index, Y, U)   
    Pf <- df %>% select(Index, Y, p_y) 
    Sf <- df %>% select(Index, Y, sig_hat) 
    
    names(MFf )[3]  <-  namehere 
    names(Lf )[3]  <-  namehere 
    names(Uf )[3]  <-  namehere 
    names(Pf )[3]  <-  namehere 
    names(Sf )[3]  <-  namehere 
  }
  
  if(i == 1){
    metaFrame <- MFf
    metaL <-Lf
    metaU <-Uf
    metaP <-Pf
    metaS <- Sf
  }else{
    metaFrame  <- left_join( metaFrame , MFf %>% select(-Y), by="Index"  )
    metaL  <-  left_join( metaL , Lf %>% select(-Y), by="Index"  )
    metaU  <-  left_join( metaU , Uf %>% select(-Y), by="Index"  )
    metaP <- left_join( metaP , Pf %>% select(-Y), by="Index"  )
    metaS <- left_join( metaS , Sf %>% select(-Y), by="Index"  )
  }
  
}



#metaL[metaL<0] = 0

m1_test_yhat <-  metaFrame  # modify
m1_test_L <- metaL  # modify
m1_test_U <- metaU   # modify

metalist$m1_test_yhat = m1_test_yhat
metalist$m1_test_L = m1_test_L
metalist$m1_test_U = m1_test_U
metalist$m1_test_py = metaP
metalist$m1_test_sig = metaS
# 12 month test MF
# out: three df: metaFrame, metaL, metaU

setwd(paste(mypath, "Py3_sim/Out_test_MF/month12/", sep ='')) # modify
file_ls <- list.files(path= paste(mypath,  "Py3_sim/Out_test_MF/month12/", sep =''), pattern=".feather$") # modify

for (i in 1:length(file_ls)){
  
  df = read_feather(file_ls[i])
  namehere = file_ls[i] %>% str_remove("_12m_dist.feather")  # modify
  
  if(sum(names(df) %in% c('L',"U") )== 0){
    
    # no L U, then generate one use sig_hat
    
    MFf <- df %>% select(Index, Y, y_hat)   
    Lf <-df %>% mutate(L = qnorm(alpha/2, y_hat, sig_hat)) %>% select(Index, Y, L)
    Uf <-df %>% mutate(U = qnorm(1-alpha/2, y_hat, sig_hat)) %>% select(Index, Y, U)   
    Pf <- df %>% select(Index, Y, p_y) 
    Sf <- df %>% select(Index, Y, sig_hat) 
    
    names(MFf )[3]  <-  namehere 
    names(Lf )[3]  <-  namehere 
    names(Uf )[3]  <-  namehere 
    names(Pf )[3]  <-  namehere 
    names(Sf )[3]  <-  namehere 
    
  }else{
    
    # have L U
    
    MFf <- df %>% select(Index, Y, y_hat)   
    Lf <-df %>% select(Index, Y, L)   
    Uf <-df %>% select(Index, Y, U)   
    Pf <- df %>% select(Index, Y, p_y) 
    Sf <- df %>% select(Index, Y, sig_hat) 
    
    names(MFf )[3]  <-  namehere 
    names(Lf )[3]  <-  namehere 
    names(Uf )[3]  <-  namehere 
    names(Pf )[3]  <-  namehere 
    names(Sf )[3]  <-  namehere 
  }
  
  if(i == 1){
    metaFrame <- MFf
    metaL <-Lf
    metaU <-Uf
    metaP <-Pf
    metaS <- Sf
  }else{
    metaFrame  <- left_join( metaFrame , MFf %>% select(-Y), by="Index"  )
    metaL  <-  left_join( metaL , Lf %>% select(-Y), by="Index"  )
    metaU  <-  left_join( metaU , Uf %>% select(-Y), by="Index"  )
    metaP <- left_join( metaP , Pf %>% select(-Y), by="Index"  )
    metaS <- left_join( metaS , Sf %>% select(-Y), by="Index"  )
  }
  
}

#metaL[metaL<0] = 0

m12_test_yhat <-  metaFrame  # modify
m12_test_L <- metaL  # modify
m12_test_U <- metaU   # modify

metalist$m12_test_yhat = m12_test_yhat
metalist$m12_test_L = m12_test_L
metalist$m12_test_U = m12_test_U
metalist$m12_test_py = metaP
metalist$m12_test_sig = metaS
length(metalist)

mypath ="C:/Users/Yi/Documents/Research/Ah_Research/WaterUsage/"
setwd(paste(mypath, "Py3_sim/Ensemble/", sep ='')) 

save(metalist,file='metalist.rda')
#             metalist


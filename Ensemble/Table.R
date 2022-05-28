#                     rm(list=ls()); sapply(dev.list(), dev.off); cat('\014')

library(tidyverse)
library(magrittr)
library(caret)
library(feather)
library(mgcv)
library(glmnet)
library(nnls)
#library(CVXR)

mypath ="C:/Users/Yi/Documents/Research/Ah_Research/WaterUsage/"
setwd(mypath)
source("mysource.R")
load('WU.rda')
setwd(paste(mypath, "Py3/Ensemble/", sep ='')) # modify
load('metalist.rda')


# 1 month
# NLL
-log(metalist$m1_test_py %>% select(-Index,-Y)) %>% map_dbl(mean) %>% round(.,2)


# 12 month
# NLL
-log(metalist$m12_test_py %>% select(-Index,-Y)) %>% map_dbl(mean) %>% round(.,2)

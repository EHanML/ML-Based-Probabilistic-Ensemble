#                     rm(list=ls()); sapply(dev.list(), dev.off); cat('\014')

library(tidyverse)
library(magrittr)
library(caret)
library(arrow)
library(mgcv)
library(glmnet)
library(nnls)
library(ggrepel)
#library(CVXR)

mypath ="C:/Users/Yi/Documents/Research/Ah_Research/WaterUsage/"
setwd(mypath)
source("mysource.R")
load('WU.rda')
setwd(paste(mypath, "Py3_sim/Ensemble/", sep ='')) # modify
load('metalist.rda')


setwd(paste(mypath, "Py3_sim/Ensemble/", sep ='')) # modify


month1_stack <- metalist$m1_test_yhat
month12_stack <- metalist$m12_test_yhat

STinfo <- WU %>% select(Index, year, month, monthIndex, ParcelID)

month1_df <- left_join(month1_stack, STinfo, by='Index') %>% select(Index, year, month, monthIndex, ParcelID, everything())
month12_df <- left_join(month12_stack, STinfo, by='Index') %>% select(Index, year, month, monthIndex, ParcelID, everything())



# monthly RMSE performance

df = month1_df 

ModelsNames <- names(df   %>% select(-Index,-year,-month,-Y, -monthIndex,-ParcelID))
monthwiseResult <- mk_outframe(c("monthIndex",ModelsNames ))

uniqueMonth <- df$monthIndex %>% unique() %>%length() 
uniqueMonthpool <-df$monthIndex %>% unique() 

# loop month
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    rslts <- df %>% filter(monthIndex == monthhere)
    
    # RMSE
    monthwiseResult[im, Mhere] <-  getRMSE(rslts $Y, rslts  %>% select( !!rlang::sym(Mhere)) %>% unlist )
    
    
  }
  
}




# plot for paper 800X400

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x=NULL)+
  labs(y="RMSE")+
  xlim(134,146)+
  theme(legend.position="none",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p


{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month1_aggregated.png" ),
      width =1300, height =400)
  
  print( p)
  dev.off()
  }



#### Uncertainty ###
# AWPI
Ldf = metalist$m1_test_L # m
Udf = metalist$m1_test_U # m

AWPI_matrix <- (Udf -Ldf) %>% select(-Index,-Y) 

df1 <- bind_cols( Ldf %>% select(Index,Y), AWPI_matrix ) 

df <- left_join(df1 , STinfo, by='Index') %>% select(Index, year, month, monthIndex, ParcelID, everything())

ModelsNames <- names(df   %>% select(-Index,-year,-month,-Y, -monthIndex,-ParcelID))
monthwiseResult <- mk_outframe(c("monthIndex",ModelsNames ))

uniqueMonth <- df$monthIndex %>% unique() %>%length() 
uniqueMonthpool <-df$monthIndex %>% unique() 

# loop month
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    rslts <- df %>% filter(monthIndex == monthhere)
    
    # RMSE
    monthwiseResult[im, Mhere] <- mean( rslts  %>% select( !!rlang::sym(Mhere)) %>% unlist )
    
    
  }
  
}


# plot for paper 800X400


picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x=NULL)+
  labs(y="AWPI")+
  xlim(134,146)+
  theme(legend.position="none",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p



{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month1_aggregated_awpi.png" ),
      width =1300, height =400)
  
  print( p)
  dev.off()
  }


#ECPI

# loop month
monthwiseResult <- mk_outframe(c("monthIndex",ModelsNames ))
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    
    mLdf <- bind_cols( Ldf , df %>% select( monthIndex)) %>% filter(monthIndex == monthhere)
    mUdf <- bind_cols( Udf , df %>% select( monthIndex)) %>% filter(monthIndex == monthhere)
    
    bool = (mLdf %>% select( !!rlang::sym(Mhere)) <= mLdf$Y) & (mUdf %>% select( !!rlang::sym(Mhere)) >= mLdf$Y)
    mean(bool)
    
    # RMSE
    monthwiseResult[im, Mhere] <- mean(bool)
    
    
  }
  
}


# plot for paper 800X400

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x="Month Index",y="ECPI")+
  xlim(134,146)+
  theme(legend.position="bottom",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        legend.title = element_blank(),
        legend.text = element_text(size=20),
        legend.background = element_rect(fill="snow1",size = 1, colour = "black"),
        legend.spacing.x = unit(1,'cm'),
        legend.key = element_blank(),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.x = element_text(size =30, face = 'bold',color="chocolate3"),
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
        geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),show.legend = FALSE,
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p


{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month1_aggregated_ecpi.png" ),
      width =1300, height =510)
  
  print( p)
  dev.off()
}



#NOIS

# loop month
monthwiseResult <- mk_outframe(c("monthIndex", ModelsNames ))
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    
    mLdf <- bind_cols( Ldf , df %>% select( monthIndex)) %>% filter(monthIndex == monthhere)
    mUdf <- bind_cols( Udf , df %>% select( monthIndex)) %>% filter(monthIndex == monthhere)
    
    L = mLdf %>% select( !!rlang::sym(Mhere)) %>% unlist
    U = mUdf %>% select( !!rlang::sym(Mhere)) %>% unlist
    alpha = 0.05
    y =  mLdf$Y %>% unlist
    awpi = mean(U-L)
    
    nois = awpi + (2/alpha) * ( sum((L - y)[L > y])  +  sum((y - U)[y > U])  )/length(y)
    
    # RMSE
    monthwiseResult[im, Mhere] <-    nois
    
    
  }
  
}

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x=NULL)+
  labs(y="NOIS")+
  xlim(134,146)+
  theme(legend.position="none",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p


{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month1_aggregated_nois.png" ),
      width =1300, height =400)
  
  print( p)
  dev.off()
}


# NLL

df1 = metalist$m1_test_py


df <- left_join(df1 , STinfo, by='Index') %>% select(Index, year, month, monthIndex, ParcelID, everything())

ModelsNames <- names(df   %>% select(-Index,-year,-month,-Y, -monthIndex,-ParcelID))
monthwiseResult <- mk_outframe(c("monthIndex",ModelsNames ))

uniqueMonth <- df$monthIndex %>% unique() %>%length() 
uniqueMonthpool <-df$monthIndex %>% unique() 


# loop month
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    rslts <- df %>% filter(monthIndex == monthhere)
    
    # RMSE
    monthwiseResult[im, Mhere] <-  -mean(log(rslts  %>% select( !!rlang::sym(Mhere)) %>% unlist)  ) 
    
    
  }
  
}


# plot for paper 800X400

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x=NULL)+
  labs(y="NLL")+
  xlim(134,146)+
  theme(legend.position="none",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p

{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month1_aggregated_nll.png" ),
      width =1300, height =400)
  
  print( p)
  dev.off()
}



###################################################################################################################################################


# 12 monthly performance

df = month12_df 

ModelsNames <- names(df   %>% select(-Index,-year,-month,-Y, -monthIndex,-ParcelID))
monthwiseResult <- mk_outframe(c("monthIndex",ModelsNames ))

uniqueMonth <- df$monthIndex %>% unique() %>%length() 
uniqueMonthpool <-df$monthIndex %>% unique() 

# loop month
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    rslts <- df %>% filter(monthIndex == monthhere)
    
    # RMSE
    monthwiseResult[im, Mhere] <-  getRMSE(rslts $Y, rslts  %>% select( !!rlang::sym(Mhere)) %>% unlist )
    
    
  }
  
}


# plot for paper 800X400

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x=NULL)+
  labs(y="RMSE")+
  xlim(134,146)+
  theme(legend.position="none",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p


{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month12_aggregated.png" ),
      width =1300, height =400)
  
  print( p)
  dev.off()
  }



#### Uncertainty ###
# AWPI
Ldf = metalist$m12_test_L # m
Udf = metalist$m12_test_U # m

AWPI_matrix <- (Udf -Ldf) %>% select(-Index,-Y) 

df1 <- bind_cols( Ldf %>% select(Index,Y), AWPI_matrix ) 

df <- left_join(df1 , STinfo, by='Index') %>% select(Index, year, month, monthIndex, ParcelID, everything())

ModelsNames <- names(df   %>% select(-Index,-year,-month,-Y, -monthIndex,-ParcelID))
monthwiseResult <- mk_outframe(c("monthIndex",ModelsNames ))

uniqueMonth <- df$monthIndex %>% unique() %>%length() 
uniqueMonthpool <-df$monthIndex %>% unique() 

# loop month
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    rslts <- df %>% filter(monthIndex == monthhere)
    
    # RMSE
    monthwiseResult[im, Mhere] <- mean( rslts  %>% select( !!rlang::sym(Mhere)) %>% unlist )
    
    
  }
  
}


# plot for paper 800X400

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x=NULL)+
  labs(y="AWPI")+
  xlim(134,146)+
  theme(legend.position="none",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p

{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month12_aggregated_awpi.png" ),
      width =1300, height =400)
  
  print( p)
  dev.off()
}


#ECPI

# loop month
monthwiseResult <- mk_outframe(c("monthIndex",ModelsNames ))
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    
    mLdf <- bind_cols( Ldf , df %>% select( monthIndex)) %>% filter(monthIndex == monthhere)
    mUdf <- bind_cols( Udf , df %>% select( monthIndex)) %>% filter(monthIndex == monthhere)
    
    bool = (mLdf %>% select( !!rlang::sym(Mhere)) <= mLdf$Y) & (mUdf %>% select( !!rlang::sym(Mhere)) >= mLdf$Y)
    
    # RMSE
    monthwiseResult[im, Mhere] <- mean(bool)
    
    
  }
  
}


# plot for paper 800X400

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x="Month Index",y="ECPI")+
  xlim(134,146)+
  theme(legend.position="bottom",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        legend.title = element_blank(),
        legend.text = element_text(size=20),
        legend.background = element_rect(fill="snow1",size = 1, colour = "black"),
        legend.spacing.x = unit(1,'cm'),
        legend.key = element_blank(),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.x = element_text(size =30, face = 'bold',color="chocolate3"),
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),show.legend = FALSE,
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p

{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month12_aggregated_ecpi.png" ),
      width =1300, height =510)
  
  print( p)
  dev.off()
}



#NOIS

# loop month
monthwiseResult <- mk_outframe(c("monthIndex", ModelsNames ))
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    
    mLdf <- bind_cols( Ldf , df %>% select( monthIndex)) %>% filter(monthIndex == monthhere)
    mUdf <- bind_cols( Udf , df %>% select( monthIndex)) %>% filter(monthIndex == monthhere)
    
    L = mLdf %>% select( !!rlang::sym(Mhere)) %>% unlist
    U = mUdf %>% select( !!rlang::sym(Mhere)) %>% unlist
    alpha = 0.05
    y =  mLdf$Y %>% unlist
    awpi = mean(U-L)
    
    nois = awpi + (2/alpha) * ( sum((L - y)[L > y])  +  sum((y - U)[y > U])  )/length(y)
    
    # RMSE
    monthwiseResult[im, Mhere] <-    nois
    
    
  }
  
}

# plot for paper 800X400

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x=NULL)+
  labs(y="NOIS")+
  xlim(134,146)+
  theme(legend.position="none",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p

{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month12_aggregated_nois.png" ),
      width =1300, height =400)
  
  print( p)
  dev.off()
}


# NLL


# NLL

df1 = metalist$m12_test_py


df <- left_join(df1 , STinfo, by='Index') %>% select(Index, year, month, monthIndex, ParcelID, everything())

ModelsNames <- names(df   %>% select(-Index,-year,-month,-Y, -monthIndex,-ParcelID))
monthwiseResult <- mk_outframe(c("monthIndex",ModelsNames ))

uniqueMonth <- df$monthIndex %>% unique() %>%length() 
uniqueMonthpool <-df$monthIndex %>% unique() 


# loop month
for(im in 1:uniqueMonth ){
  
  monthhere <- uniqueMonthpool[im] 
  monthwiseResult[im,"monthIndex"] <-    monthhere 
  
  # loop models
  for(m in 1:length(ModelsNames)){
    
    Mhere <- ModelsNames[m] 
    
    rslts <- df %>% filter(monthIndex == monthhere)
    
    # RMSE
    monthwiseResult[im, Mhere] <-  -mean(log(rslts  %>% select( !!rlang::sym(Mhere)) %>% unlist)  ) 
    
    
  }
  
}


# plot for paper 800X400

picdf <- monthwiseResult %>% 
  mutate_at(vars(matches("MonthIndex")), ~(as.integer(.)) )%>%
  gather(ModelsNames,key="Model",value="RMSE")

p<- picdf %>%
  ggplot(mapping = aes(x=monthIndex, y= RMSE, color = Model))+
  geom_point(size=5)+
  geom_line(size=1.2,alpha=0.5 ) + # use the code below rather than this 
  #ggtitle(sprintf("1-Month Ahead Forecasting RMSE Peformance"))+
  labs(x=NULL)+
  labs(y="NLL")+
  xlim(134,146)+
  theme(legend.position="none",  panel.background = element_rect(fill = "white",color="black"), 
        panel.grid.major = element_line(size = 1, linetype = 'solid',colour = "wheat1"),
        axis.text.x = element_text(size = 15,color="chocolate3"), axis.text.y = element_text(size  = 15,color='firebrick3'), 
        axis.title.y = element_text(size =30, face = 'bold',color='firebrick3'))+
  geom_label_repel(data = picdf %>% filter(monthIndex == max(monthIndex)),
                   mapping=aes(x=monthIndex, y=RMSE,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'y')
p


{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("month12_aggregated_nll.png" ),
      width =1300, height =400)
  
  print( p)
  dev.off()
}




############# Error anaylsis ################
library(latex2exp)

  
  dist_pic_generator <- function(modelnames,df,force = 10){
    MAP <- as_tibble(mk_outframe(c('Model','x','y')))[-1,] 
    for(n in modelnames){
      dense <- density(df %>% gather(modelnames,key="Model",value = "Y_hat") %>% mutate(res = Y-Y_hat) %>% filter(Model == n) %$% res )
      ith <- which.max(dense$y)
      x <- dense$x[ith]; y <- dense$y[ith]
      MAPith <- tibble(Model = n, x = x, y = y)
      MAP <- bind_rows(MAP,MAPith)
    }
    MAP
    pic<- df %>% gather(modelnames,key="Model",value = "Y_hat") %>%
      ggplot()+
      geom_density(mapping = aes(x = Y -  Y_hat, fill = Model),color='black',alpha = 0.5) + 
      xlim(-10000,10000)+
      ylim(0,0.0013)+
      labs(x=TeX("$y - \\hat{y}$"))+
      theme_light()+
      geom_label_repel(data = MAP,
                       mapping=aes(x=x, y=y,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'x', force = force)
    return(pic)
  }
  
  
  #########################################################
  
  sn <- month1_df %>% select(ModelsNames) %>% select(-contains('ing')) %>% names
  
  en <- month1_df %>% select(ModelsNames) %>% select(contains('ing')) %>% names
  # ensemble 
  pic <- dist_pic_generator(en, month1_df,force =30)
  pic 
  {
    setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
    png(filename = sprintf("dist_en_prd_m1.png" ),
        width =1400, height =400)
    
    print( pic)
    dev.off()
  }
  
  # individual 
  pic <- dist_pic_generator(sn, month1_df,force = 30)
  pic 
  {
    setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
    png(filename = sprintf("dist_cl_prd_m1.png" ),
        width =1400, height =400)
    
    print( pic)
    dev.off()
  }
  

  

  
  ### QQ ###
  
  dd<- month1_df %>% gather(ModelsNames,key="Model",value = "Y_hat") %>% mutate(res = Y-Y_hat) 
  ModelsNames
  
  out <- tibble(Model = NA,QM=NA,QN=NA)
  out <- out[-1,]
  for(i in ModelsNames){
    
    resi <- dd %>% filter(Model == i) %$% res
    sig <- sd(resi)
    Qm <- quantile( resi , probs = seq(0.001, 0.999, 0.001)) %>% unlist
    Qn <- qnorm(seq(0.001, 0.999, 0.001),0,sig)
    outi <- tibble(Model = i,QM=Qm,QN=Qn)
    out <- bind_rows(out,outi)
  }
  pic<- out %>% 
    ggplot()+
    geom_point(mapping = aes(x=QN,
                             y =QM,color = Model ),size=1.5,alpha=0.7) +
    
    geom_abline(intercept = 0, slope = 1, color="black", 
                linetype="dashed", size=1.5)+
    labs(x='Normal', y='Model')+
    theme_light()
  
  pic
  {
    setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
    png(filename = sprintf("QQ.png" ),
        width =800, height =600)
    
    print( pic)
    dev.off()
    }
  
  
  
  ### QQ ###
  
  # Even & odd
  
  month1_df %>% filter(month%%2 == 0)%>%
    gather(ModelsNames,key="Model",value = "Y_hat") %>%
    ggplot()+
    ylim(0,0.002)+
    geom_density(mapping = aes(x = Y -  Y_hat, fill = Model),color='black',alpha = 0.5) + 
    labs(x='Residual',title = 'Even months')+

    theme_light()
  
  month1_df %>% filter(month%%2 != 0)%>%
    gather(ModelsNames,key="Model",value = "Y_hat") %>%
    ggplot()+
    ylim(0,0.002)+
    geom_density(mapping = aes(x = Y -  Y_hat, fill = Model),color='black',alpha = 0.5) + 
    labs(x='Residual',title = 'Odd months')+

    theme_light()
  
  month1_df %>% filter(month%%2 == 0)%>%
    gather(sn,key="Model",value = "Y_hat") %>%
    ggplot()+
    geom_density(mapping = aes(x = Y -  Y_hat, fill = Model),color='black',alpha = 0.5) + 
    labs(x='Residual')+

    theme_light()
  
  month1_df %>% filter(month%%2 != 0)%>%
    gather(sn,key="Model",value = "Y_hat") %>%
    ggplot()+
    geom_density(mapping = aes(x = Y -  Y_hat, fill = Model),color='black',alpha = 0.5) + 
    labs(x='Residual')+
    theme_light()
  
  month1_df %>% filter(month%%2 == 0)%>%
    gather(en,key="Model",value = "Y_hat") %>%
    ggplot()+
    geom_density(mapping = aes(x = Y -  Y_hat, fill = Model),color='black',alpha = 0.5) + 
    labs(x='Residual')+
    theme_light()
  
  month1_df %>% filter(month%%2 != 0)%>%
    gather(en,key="Model",value = "Y_hat") %>%
    ggplot()+
    geom_density(mapping = aes(x = Y -  Y_hat, fill = Model),color='black',alpha = 0.5) + 
    labs(x='Residual')+

    theme_light()
  
  
  
  


metalist$m1_test_sig







# month 12
############# Error anaylsis ################
library(latex2exp)


dist_pic_generator <- function(modelnames,df,force = 10){
  MAP <- as_tibble(mk_outframe(c('Model','x','y')))[-1,] 
  for(n in modelnames){
    dense <- density(df %>% gather(modelnames,key="Model",value = "Y_hat") %>% mutate(res = Y-Y_hat) %>% filter(Model == n) %$% res )
    ith <- which.max(dense$y)
    x <- dense$x[ith]; y <- dense$y[ith]
    MAPith <- tibble(Model = n, x = x, y = y)
    MAP <- bind_rows(MAP,MAPith)
  }
  MAP
  pic<- df %>% gather(modelnames,key="Model",value = "Y_hat") %>%
    ggplot()+
    geom_density(mapping = aes(x = Y -  Y_hat, fill = Model),color='black',alpha = 0.5) + 
    xlim(-10000,10000)+
    ylim(0,0.00045)+
    labs(x=TeX("$y - \\hat{y}$"))+
    theme_light()+
    geom_label_repel(data = MAP,
                     mapping=aes(x=x, y=y,label=Model),nudge_x = 1,size= 5,na.rm = TRUE,direction = 'x', force = force)
  return(pic)
}


#########################################################

sn <- month12_df %>% select(ModelsNames) %>% select(-contains('ing')) %>% names

en <- month12_df %>% select(ModelsNames) %>% select(contains('ing')) %>% names
# ensemble 
pic <- dist_pic_generator(en, month12_df,force =30)
pic 
{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("dist_en_prd_m12.png" ),
      width =1400, height =400)
  
  print( pic)
  dev.off()
}

# individual 
pic <- dist_pic_generator(sn, month12_df,force = 30)
pic 
{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("dist_cl_prd_m12.png" ),
      width =1400, height =400)
  
  print( pic)
  dev.off()
}





### QQ ###

dd<- month12_df %>% gather(ModelsNames,key="Model",value = "Y_hat") %>% mutate(res = Y-Y_hat) 
ModelsNames

out <- tibble(Model = NA,QM=NA,QN=NA)
out <- out[-1,]
for(i in ModelsNames){
  
  resi <- dd %>% filter(Model == i) %$% res
  sig <- sd(resi)
  Qm <- quantile( resi , probs = seq(0.001, 0.999, 0.001)) %>% unlist
  Qn <- qnorm(seq(0.001, 0.999, 0.001),0,sig)
  outi <- tibble(Model = i,QM=Qm,QN=Qn)
  out <- bind_rows(out,outi)
}
pic<- out %>% 
  ggplot()+
  geom_point(mapping = aes(x=QN,
                           y =QM,color = Model ),size=1.5,alpha=0.7) +
  
  geom_abline(intercept = 0, slope = 1, color="black", 
              linetype="dashed", size=1.5)+
  labs(x='Normal', y='Model')+
  theme_light()

pic
{
  setwd(paste(mypath,  "Py3/Ensemble/plots/", sep =''))
  png(filename = sprintf("QQ12.png" ),
      width =800, height =600)
  
  print( pic)
  dev.off()
}



### QQ ###
























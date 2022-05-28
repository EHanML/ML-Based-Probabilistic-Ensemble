
library(tidyverse)
library(caret)




# seed = sample.int(1000, 1)
###################################
mk_outframe <- function(col_name){
  length_name <- length(col_name)
  models  <- data.frame(matrix(rep(NA, length_name), nrow = 1))
  names(models) <- col_name 
  return(models)
}

###################################
addLag <- function(data,    # data frame # contains time series
                   time,     # integer sequence !!! #
                   group,     # single col name #
                   features,   # array of col names # feature we want to have lag in the result
                   nlag,       # single integer # how many lag we want to generate
                   lagRsl = 1) # time resolution  
{
  if(nlag == 0){
    finalout <- data
  }else{
    
    
    data <- as_tibble(data)
    lg <- nlag
    
    # summary each group start time and end time
    lagsummary <- data %>% 
      group_by(!! rlang::sym(group)) %>%  summarize(st = min(!! rlang::sym(time)), ft = max(!! rlang::sym(time))) 
    
    # for each group, create [full time point tibble] in order to get NAs
    out <- tibble()
    
    for(i in 1:dim(lagsummary)[1]) {
      out2 <- tibble( Group = lagsummary[i,group][[1]], fullTime = seq(lagsummary[i,"st"][[1]], lagsummary[i,"ft"][[1]], by = 1))
      out <- bind_rows(out, out2)
    }
    names(out) <- c(group, time)
    
    # merge response with time 
    full <- merge(data, out, by = c(group, time) , all.y = T) %>% arrange(!! rlang::sym(group),!! rlang::sym(time)) %>% as_tibble()
    
    #here we add lags for each feature
    lagst <- full %>% transmute(Location = !! rlang::sym(group))
    for(i in 1: nlag ){ # for 1 - nlag
      for(j in 1:length(features)){ # for each feature to be lagged
        
        onelag <-  full %>% group_by(!! rlang::sym(group)) %>%
          transmute( LAG =  lag(!! rlang::sym(features[j]), i) )  %>% ungroup()
        
        lagst <- bind_cols(lagst, onelag[,2])
        
      }
      
    }
    
    lagst <- select(lagst, -1)
    
    colnames(lagst) <- sprintf("%s_lag%d",
                               rep(features,lg), 
                               rep(1:lg,rep(length(features),lg)))
    
    dim(lagst) - dim(full)
    
    finalout <- bind_cols(full, lagst)
  }
  
  return(finalout)
}

###################################
chck_factors <- function(dataset){
  out1 <- mk_outframe(c("Feature","Type","Example","Levels"))
  lvl1 <- c()
  dataset %>% select_if(~!is.double(.)) -> nonDBL # select_if double
  for(p in 1:dim(nonDBL )[2]){
    nonDBL %>% select(p) %>% as.matrix() %>% unique %>% length -> lvl
    nonDBL %>% select(p) %>% .[1,1] %>% as.character ->eg
    nonDBL %>% select(p) %>% unlist() %>% typeof  -> typ
    nonDBL %>% select(p) %>% names  -> nm
   
    out1[p,"Feature"] <- nm
    out1[p,"Type"] <- typ
    out1[p,"Example"] <- eg
    out1[p,"Levels"] <- lvl
    
    if(lvl == 1) lvl1 <- c(lvl1,nm)
  }
  return( list(summary = out1, LevelOne = lvl1))
}
###################################

add_spline <- function(dframe,certain_part, spline_type, same_part){
  all.var <- names(  dframe )
  if(length(grep(same_part, all.var)) == 0){
    
    sf <- as.formula(certain_part)
    
  }else{
    
    add_spline <- all.var[grep(same_part, all.var)]
    uncertain_part <- paste("+ ", spline_type, "(", add_spline, ")", sep = "", collapse=" " )
    sf <- paste(certain_part, uncertain_part, sep="")
    sf <- as.formula(sf)
    
  }
  return(sf)
}

###################################

GroupTimeSlices <- function(data,  # grouped time series data frame (train data) # TEST:  data=s.train
                            TrainPortion = 0.5,  # [0,1]
                            K=5, # how many folds for rolling partition training: affect computational speed!
                            FixedWindow = FALSE,
                            groupTimeIndex){ # split according to this time index # TEST:  groupTimeIndex="monthIndex"
  
  timeIndex <- data %>% select(!! rlang::sym(groupTimeIndex)) %>% unique %>% pull
  indexFrame <- tibble(DatatimeIdex = timeIndex, caretTimeIndex = 1:length(timeIndex))
  remainder <- (length(timeIndex) - floor(length(timeIndex) * TrainPortion)) %% K
  InitialWindow <- floor(length(timeIndex) * TrainPortion)  + remainder
  Horizon_n_Skip <- (length(timeIndex) - InitialWindow )  %/% K
  message(sprintf("Horizon and Skip are equal to %d",Horizon_n_Skip))
  
  caretFold <- createTimeSlices(indexFrame$caretTimeIndex, initialWindow = InitialWindow , fixedWindow = FixedWindow ,
                                horizon = Horizon_n_Skip ,skip =Horizon_n_Skip-1) 
  
  s.trainhere <- data %>% mutate(Indexhere = 1:dim(data)[1]) 
  
  indexFit <- list()
  indexPrd <- list()
  
  for(k in 1:K){
    groupIndexFit.k <- filter(indexFrame, caretTimeIndex %in% caretFold$train[[k]])$DatatimeIdex
    groupindexPrd.k <- filter(indexFrame, caretTimeIndex %in% caretFold$test[[k]])$DatatimeIdex
    indexFit[[k]] <- filter(s.trainhere, !! rlang::sym(groupTimeIndex) %in% groupIndexFit.k)$Indexhere %>% sort()
    indexPrd[[k]] <- filter(s.trainhere, !! rlang::sym(groupTimeIndex) %in% groupindexPrd.k)$Indexhere %>% sort()
  }
  
  if(TrainPortion != 0){
    message("It will be a rolling forecast index!")
    message("==> Call Index4Fit & Index4Prd")
    return(list(Index4Fit =  indexFit , Index4Prd =  indexPrd  ))
    
  }else{
    message("It will be a time slice index!")
    message("==> Call Index4Prd")
    indexPrd[[1]] <- c( indexFit[[1]],indexPrd[[1]] ) %>% unique() %>% sort()
    return(list(Index4Prd =  indexPrd  ))
  }
  
  
}


###################################
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df
  df$random = (1:nrow(df))/nrow(df)
  df
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  
  return(sum(df$Gini))
}
###################################
NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

###################################

NOIS = function(Y.obs, #N-vector of predicted values
                Y.int, #Nx2 matrix, 1st col lower, 2nd col upper, can be NA for 1-sided intervals
                sided = "upper", #one of 'center', 'upper', what type of CI
                alpha = 0.05){ #alpha for the interval ({1-alpha}100% for one-sided,  {1-alpha/2}100% for 'center')
  
  if(sided == "center"){
    
    out = mean( (Y.int[ ,2] - Y.int[ ,1]) + 
                  (2 / alpha) * (Y.int[ ,1] - Y.obs) * (Y.int[ ,1] > Y.obs) + 
                  (2 / alpha) * (Y.obs - Y.int[ ,2]) * (Y.int[ ,2] < Y.obs) )
    
  } else if(sided == "upper"){
    
    out = mean(Y.int[ ,2] + (1 / alpha) * (Y.obs - Y.int[ ,2]) * (Y.int[ ,2] < Y.obs) )
    
  } else stop("'sided' must be one of 'center', 'upper'")
  
  return(out)
}

###################################
# RMSE
getRMSE <- function(obs, prd){
  sqrt(mean((prd - obs)^2)) 
}

###################################
# MER
getMER <- function(obs, prd){
  
  cf <- data.frame(cbind((prd == 0), (obs == 0)))
  if(dim(table(cf))[1] == 2){
    MER <- 1- (table(cf)[1,1] + table(cf)[2,2])/(sum(table(cf)))
  }else{
    MER <- 1- (table(cf)[1,1])/(sum(table(cf)))
  }
  return(MER)
}

# F
###################################
harmonic <- function(v){ 
  length(v)/sum(1/v)
}


###################################
getFbeta <- function(obs, prd, alpha = 1 , beta = 1){  #   beta>1 Recall impact more; beta<1 Precision impact more
  
  frame <- tibble(OBS= obs, PRD = prd)
  metric_frame <- frame %>% mutate(MIN = pmin(OBS,PRD),
                                   P = (MIN+alpha)/(PRD+alpha),
                                   R = (MIN+alpha)/(OBS+alpha),
                                   Fbeta = 1/(  (1/(1+beta^2)) * (1/P + (beta^2)/R)      )
  )        # %>% print(n=100)
  
  macro_Fbeta <- 1/(  (1/(1+beta^2)) * (1/mean(metric_frame$P) + (beta^2)/mean(metric_frame$R))      )
  mean_Fbeta <- mean(metric_frame $Fbeta)
  harm_Fbeta <- harmonic(metric_frame $Fbeta)
  return(  list(metric_frame, macro_Fbeta ,mean_Fbeta , harm_Fbeta) )
}


###################################
getPrescreen <- function(caretimp, cut=70 ){
  
  impframe <- tibble(feature = rownames(caretimp$importance), Importance = caretimp$importance$Overall) %>% 
    arrange(desc(Importance)) 
  
  keep <- impframe %>% 
    filter(Importance > cut) 
  
  discard <- impframe %>% 
    filter(Importance <= cut) 
  
  return(list(keep =   keep$feature, discard = discard$feature))
  
}

# getPrescreen(varImp(Fit1),70)

###################################

getCorRm <- function(data, name_contain){
  
  dmrdct <- WU %>% select(contains(name_contain))
  keepname <- preProcess(dmrdct, method="corr") %>% predict(., dmrdct) %>%names
  discardnames1 <- dmrdct %>% select(contains(name_contain)) %>% select(-keepname) %>%names
  
  return( discardnames1)
}

###################################

library(Rtsne)
getTSNE <- function(  data, target, perplexity=50, theta=0.5, dim = 2){
  
  m <- data %>% select(-!! rlang::sym(target)) # remove y
  m2 <- preProcess(m, method= c("center","scale") ) %>% predict(., m) # normalize
  tsne_model_1 = Rtsne(as.matrix(m2), check_duplicates=FALSE, pca=TRUE, perplexity=perplexity, theta=theta, dims= dim)
  
  d_tsne_1 <- cbind(data %>% select(!! rlang::sym(target)), as.data.frame(tsne_model_1$Y ) ) # map here 
  
                sprintf("D%d", rep(1:dim))
                
  
  names( d_tsne_1) <- c("target" , sprintf("D%d", rep(1:dim)))
  
  return(d_tsne_1)
  
}
  
  
  getTSNEplot <- function(data,  perplexity=50, typeCtn = T, legend = F){ 
    
    d_tsne_1 <- data
  if(typeCtn == T) {
    ggplot(d_tsne_1, aes(x=D1, y=D2, colour = target ) ) +
      geom_point(size=1) +
      scale_colour_gradient(low = "orange", high = "blue" ) +
      xlab("") + ylab("") +
      ggtitle( sprintf("t-SNE perplexity = %g",perplexity) )  +
      theme(axis.text.x=element_blank(),
            axis.text.y=element_blank()) 
    
  } else {
    ggplot(d_tsne_1, aes(x=D1, y=D2, colour = target ) ) +
      geom_point(size=1) +
      xlab("") + ylab("") +
      theme(legend.position= ifelse(legend == legend ,"none", "right" )) +
      ggtitle( sprintf("t-SNE perplexity = %g",perplexity) ) +
      theme(axis.text.x=element_blank(),
            axis.text.y=element_blank()) 
    

  }
  
}


###################################

addMeanEncode <- function(data, target, group){
  
  # mean encoding
  MEmap <- data %>% group_by(!! rlang::sym(group) ) %>%
    summarize(meanY = mean(!! rlang::sym(target)), sdY = sd(!! rlang::sym(target)))
  
  DataOut <- left_join( data , MEmap ,  by=group) %>% 
    select(!! rlang::sym(target),!! rlang::sym(group) ,meanY,sdY,  everything())
  #    DataOut  %>% select(meanY,sdY,ParcelID, contains("Coord")) %>% distinct()
  
  out <- list(data = DataOut, map = MEmap)
  return(out)
}

  ###################################  
  
  ### Check missing data ###
  # feed the function a dataframe #
  
  chckmiss <- function(dataf){
    for(i in 1:dim(dataf)[2]){
      
      if(anyNA(dataf[,i])){
        
        if(table(is.na(dataf[,i]))["TRUE"] == dim(dataf)[1]){
          print(sprintf("Variable <%s> are all missing", colnames(dataf)[i]))
        }else{
          print(sprintf("Variable <%s> have missing data (NA value)", colnames(dataf)[i]))
          NAtable <- table(is.na(dataf[,i]))
          names(NAtable) = c("Number of values", "Number of NAs")
          print( NAtable)
        }
      }else{
        print(sprintf("Variable <%s> have no have missing value!", colnames(dataf)[i]))
      }
    }
  }
  
  ###################################  
  

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  ###################################  



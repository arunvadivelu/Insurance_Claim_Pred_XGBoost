rm(list=ls())
gc()
library(dplyr)
library(corrplot)
library(caret)
library(Metrics)
require(xgboost)
library(Matrix)
library(Boruta)


#Read Data:
df <- read.csv("C:\\Users\\YVASKA\\Desktop\\Data_challenges\\AnacondaCobra\\train.csv", header = T)


# Normalization:
minmax_normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

z_normalize <- function(x) {
  return ((x - mean(x)) / sd(x))
}


# numeric variables
variabl_n = c("ps_car_12","ps_car_13","ps_car_14","ps_car_15","ps_calc_01","ps_calc_02","ps_calc_03","ps_reg_01","ps_reg_02","ps_reg_03")

# Convert non-numeric to factors
nonfactor = c(variabl_n,"id","target")
df[, !colnames(df) %in% nonfactor] <- df[, !colnames(df) %in% nonfactor] %>% lapply(factor) %>% data.frame()


#Normalize the continuous Variables

df[,variabl_n] <- df %>% subset(select = variabl_n) %>% lapply(z_normalize)  %>% data.frame()
str(df)

# Check imbalance distribution
hist(subdf$target)


#Correlation for continuous variables

CR <-cor(subset(df, select = variabl_n))
corrplot(CR, method="circle")
rm(CR)


# remove the less significant numeric variables from Corplot inference
df <- df[,-which((names(df) == "ps_car_10" ) | (names(test_df) == "ps_reg_11")| (names(test_df) == "ps_reg_13") )]
str(df)


# run Boruta to evaluate the variables and get the inference

boruta.train <- Boruta(df$target~., data = df[,!(colnames(df) %in% c("id", "target"))], doTrace = 2)
saveRDS(boruta.train, file="C:\\Users\\YVASKA\\Desktop\\Data_challenges\\AnacondaCobra\\boruta_20.rda")
#newBoruta <- readRDS("C:\\Users\\YVASKA\\Desktop\\Data_challenges\\AnacondaCobra\\boruta_20.rda")

plotImpHistory(newBoruta)


# from the Boruta inference identify relavant variables and remove unwanted features (all with ps_calc_)
df <- df[, -grep("ps_calc_", colnames(df))]
variabl_n_red = c("ps_car_12","ps_car_14","ps_car_15","ps_reg_01","ps_reg_02")


factor_df<- model.matrix(~ . + 0, data=df[, !(colnames(df) %in% nonfactor)] , contrasts.arg = lapply(df[, !(colnames(df) %in% nonfactor)] , contrasts, contrasts=FALSE)) %>% data.frame()
df <- cbind(target=df$target,df[,variabl_n_red],factor_df)
rm(factor_df)

##########################################################################################################################
# Prepare Test data


test_df <- read.csv("C:\\Users\\YVASKA\\Desktop\\Data_challenges\\AnacondaCobra\\test.csv", header = T)

# Normalize
test_df[,variabl_n] <- test_df %>% subset(select = variabl_n) %>% lapply(z_normalize)  %>% data.frame()

# Convert to factors
test_df[, !(colnames(test_df) %in% nonfactor)] <- test_df[, !colnames(test_df) %in% nonfactor]  %>% lapply(factor) %>% data.frame() 
str(test_df)

# from the Boruta inference identify relavant variables and remove unwanted features (all with ps_calc_)

test_df <- test_df[, -grep("ps_calc_", colnames(test_df))]

# remove from Corplot inference
test_df <- test_df[,-which((names(df) == "ps_car_10" ) | (names(test_df) == "ps_reg_11")| (names(test_df) == "ps_reg_13") )]


factor_test_df<-model.matrix(~ . + 0, data=test_df[, !(colnames(test_df) %in% nonfactor)] , contrasts.arg = lapply(test_df[, !(colnames(test_df) %in% nonfactor)] , contrasts, contrasts=FALSE))  %>% data.frame() 

new_test_df <- cbind(test_df[,variabl_n_red],factor_test_df)
saveRDS(df, file="C:\\Users\\YVASKA\\Desktop\\Data_challenges\\AnacondaCobra\\df.rda")


####################################################################################################################
# Build the XG Boost model

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.2, gamma=0.2, max_depth=10, min_child_weight=1, subsample=1, colsample_bytree=1, early_stopping_rounds=20)
dtrain <- xgb.DMatrix(data = data.matrix(df[,!(colnames(df) %in% c("id", "target"))]), label = df$target)
xgbcv <- xgb.cv( params = params, data = dtrain,nrounds = 50, num_boost_round=100, nfold = 5, showsd = T, stratified = T, maximize = F)

# Train the model
xgbtrain <- xgb.train( params = params, data = dtrain,nrounds = 30, num_boost_round=100, nfold = 5, showsd = T, stratified = T, maximize = F)

# Evaluate the Model
mat <- xgb.importance (feature_names = colnames(df),model = xgbtrain)
xgb.plot.importance (importance_matrix = mat[1:20]) 

# Save the model locally
#bstSparse <- xgboost(data = as.matrix(df[,!(colnames(df) %in% c("id", "target"))]), label = df[,1] , max.depth = 60, eta = .3, nthread = 4, nround = 25, objective = "binary:logistic",verbose = 2)
saveRDS(xgbtrain, file="C:\\Users\\YVASKA\\Desktop\\Data_challenges\\AnacondaCobra\\xgbtrain_V1.rda")

# Predict with the test data
pred <- predict(bstSparse, as.matrix(data.frame(df[,(colnames(df) %in% c("target"))])))
pred <- data.frame(pred)

pred$id <- (test_df$id)
# distribution of the claim and no claim
hist(pred[,1])

# Convert the probabilistic values to 0 & 1
pred$result[pred$pred == max(pred$pred)] <- 1
pred$result[pred$pred == min(pred$pred)] <- 0
length(pred$pred[pred$pred == max(pred$pred)])

# Check the confusion matrix
confusionMatrix(pred$result,df$target)

# Write the predicted values to CSV
options(scipen=999)
write.csv(cbind(pred$id,pred$pred), file = "C:\\Users\\YVASKA\\Desktop\\Data_challenges\\AnacondaCobra\\submission_XGBBoruta.csv",row.names=FALSE)
#id,target


############################################################################################################################
# Test with Logistic Regression
#use cross validation
train_control<- trainControl(method="cv", number=10)

#Build Model
model<- train(target~., data=df, trControl=train_control, method="glm", family=binomial(link='logit'))
summary(model)
anova(model, test="Chisq")

###############################################################################################################

# Random Forest Model
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
rf_random <- train(target~., data=df, method="rf", metric=metric, tuneLength=15, trControl=control)

plot(rf_random)


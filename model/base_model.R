library(data.table)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(lubridate)
library(timetk)
library(tidyquant)
library(TTR)
library(zoo)
library(caret)
setwd("/Users/an-eunseog/Desktop/bike_paper/data")

edge_num_mat = fread('Edge_num_mat2.csv')
edge_num_mat %>% dim()
train = edge_num_mat[,967:991]
edge_num_mat = as.data.frame(edge_num_mat)

set.seed(1234)
control = trainControl(method = "LGOCV", p = 0.7, number = 1)


# xgb
set.seed(1234)
model_xgb = train(V271~.,
                 data = train,
                 method = "xgbTree",
                 trControl = control,
                 tuneLength = 3)

rmse = 0
mae = 0 
sd_mse = rep(0,426)
sd_mae = rep(0,426)

for (i in 1:426) {
  a = 990+i
  b = 1013+i
  c = 1014+i
  x = edge_num_mat[,a:b]
  y = edge_num_mat[,c]
  colnames(x) <- colnames(train)[1:24]
  pred = predict(model_xgb, newdata = x)
  sd_mse[i] = sqrt(mean((y-pred)^2))
  sd_mae[i] = mean(abs(y-pred))
  rmse = rmse + sqrt(mean((y-pred)^2))
  mae = mae + mean(abs(y-pred))
}

xgb_rmse = rmse/426
xgb_mae = mae/426
print(c(xgb_rmse,xgb_mae))

mat = matrix(0, 257, 426)

for (i in 1:426) {
  a = 990+i
  b = 1013+i
  c = 1014+i
  x = edge_num_mat[,a:b]
  y = edge_num_mat[,c]
  colnames(x) <- colnames(train)[1:24]
  pred = predict(model_xgb, newdata = x)
  for(j in 1:257){
    mat[j,i] = pred[j] 
  }
}

# rmse sd
xgb_mse = (mat - edge_num_mat[,1021:1440])^2
sd(sqrt(apply(xgb_mse, 1, mean))) # station sd = 0.677881
sd(sqrt(apply(xgb_mse, 2, mean))) # time sd = 0.6809052


# mae sd
xgb_mae = abs(mat - edge_num_mat[,1021:1440])
sd(apply(xgb_mae, 1, mean)) # station sd = 0.4160261
sd(apply(xgb_mae, 2, mean)) # time sd = 0.3981787



# svr
set.seed(1234)
model_svm = train(V271~.,
                 data = train,
                 method = "svmPoly",
                 trControl = control,
                 tuneLength = 4)

rmse = 0
mae = 0 
sd_mse = rep(0,426)
sd_mae = rep(0,426)


for (i in 1:426) {
  a = 990+i
  b = 1013+i
  c = 1014+i
  x = edge_num_mat[,a:b]
  y = edge_num_mat[,c]
  colnames(x) <- colnames(train)[1:24]
  pred = predict(model_svm, newdata = x)
  sd_mse[i] = sqrt(mean((y-pred)^2))
  sd_mae[i] = mean(abs(y-pred))
  rmse = rmse + sqrt(mean((y-pred)^2))
  mae = mae + mean(abs(y-pred))
}

svm_rmse = rmse/426
svm_mae = mae/426
sd(sd_mse)
sd(sd_mae)
print(c(svm_rmse,svm_mae))

mat = matrix(0, 257, 426)

for (i in 1:426) {
  a = 990+i
  b = 1013+i
  c = 1014+i
  x = edge_num_mat[,a:b]
  y = edge_num_mat[,c]
  colnames(x) <- colnames(train)[1:24]
  pred = predict(model_svm, newdata = x)
  for(j in 1:257){
    mat[j,i] = pred[j] 
  }
}

# rmse sd
svm_mse = (mat - edge_num_mat[,1021:1440])^2
sd(sqrt(apply(svm_mse, 1, mean))) # station sd = 4.358159
sd(sqrt(apply(svm_mse, 2, mean))) # time sd = 3.528991


# mae sd
svm_mae = abs(mat - edge_num_mat[,1021:1440])
sd(apply(svm_mae, 1, mean)) # station sd = 2.040994
sd(apply(svm_mae, 2, mean)) # time sd = 1.100176










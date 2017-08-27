
# Build Knn Regression model on dataset to predict income of a person . 
rm(list=ls(all=TRUE))

library("RCurl")
data=read.table(text = getURL("https://raw.githubusercontent.com/rajsiddarth119/Datasets/master/Bank_dataset.csv"), header=T, sep=',',
                col.names = c('ID', 'age', 'exp', 'inc', 
                              'zip', 'family', 'ccavg', 'edu', 
                              'mortgage', 'loan', 'securities', 
                              'cd', 'online', 'cc'))
str(data)
#Remove ID,exp and zip
data=subset(data,select = -c(ID,exp,zip))

#Converting age,inc,CCavg,Mortgage,family to numeric and standardizing
#Income as dependent variable
dep_atr=c("inc")
num_atr=c("age","ccavg","mortgage","family")
categ_atr=setdiff(names(data),c(num_atr,"inc"))

num_data=data.frame(sapply(data[num_atr], function(x){as.numeric(x)}))
library(vegan)
num_data=decostand(num_data,method = "range")

#Converting categ attributes to dummy variables
library(dummies)
categ_data=data.frame(sapply(data[categ_atr], as.factor))
categ_data=dummy.data.frame(categ_data,sep="_")

final_data=cbind(num_data,categ_data,"inc"=data$inc)
str(final_data)

# Divide the data into training,testing and eval data
set.seed(123)
library(caTools)
rowids = 1:nrow(final_data)
train_index =  sample(rowids, length(rowids)*0.6)
test_index = sample(setdiff(rowids, train_index), length(rowids)*0.2)
eval_index = setdiff(rowids, c(train_index, test_index))

train_data=final_data[train_index,]
test_data=final_data[test_index,]
eval_data=final_data[eval_index,]

# Checking how records are split with respect to target attribute.
summary(final_data$inc)
summary(train_data$inc)
summary(test_data$inc)
summary(eval_data$inc)

ind_variables=setdiff(names(final_data),"inc")
# Using Knn algorithm to predict income variable
#install.packages("FNN")
library(FNN)

# .........k = 1........................#
#Predicting on train data

pred1_train = knn.reg(train = train_data[,ind_variables],test = train_data[,ind_variables],
                     y = train_data$inc, k = 1)

pred1_test = knn.reg(train = train_data[,ind_variables],test = test_data[,ind_variables],
                    y = train_data$inc, k = 1)
#Error metrics for regression
library(DMwR)
#Train
cat("Error metrics on train data for k=1")
regr.eval(train_data[,"inc"], pred1_train$pred)
cat("MAPE for k=1 on train is",round(regr.eval(train_data[,"inc"], pred1_train$pred)[4],4)*100,"%")

#Test
cat("Error metrics on test data for k=1")
regr.eval(test_data[,"inc"], pred1_test$pred)
cat("MAPE for k=1 on test is",round(regr.eval(test_data[,"inc"], pred1_test$pred)[4],4)*100,"%")

# .........k = 3........................#
#Predicting on train data

pred3_train = knn.reg(train = train_data[,ind_variables],test = train_data[,ind_variables],
                      y = train_data$inc, k = 3)

pred3_test = knn.reg(train = train_data[,ind_variables],test = test_data[,ind_variables],
                     y = train_data$inc, k = 3)
#Error metrics for regression
#Train
cat("Error metrics on train data for k=3")
regr.eval(train_data[,"inc"], pred3_train$pred)
cat("MAPE for k=3 on train is",round(regr.eval(train_data[,"inc"], pred3_train$pred)[4],4)*100,"%")

#Test
cat("Error metrics on test data for k=3")
regr.eval(test_data[,"inc"], pred3_test$pred)
cat("MAPE for k=3 on test is",round(regr.eval(test_data[,"inc"], pred3_test$pred)[4],4)*100,"%")

# .........k = 5........................#
#Predicting on train data

pred5_train = knn.reg(train = train_data[,ind_variables],test = train_data[,ind_variables],
                      y = train_data$inc, k = 5)

pred5_test = knn.reg(train = train_data[,ind_variables],test = test_data[,ind_variables],
                     y = train_data$inc, k = 5)
#Error metrics for regression
#Train
cat("Error metrics on train data for k=5")
regr.eval(train_data[,"inc"], pred5_train$pred)
cat("MAPE for k=5 on train data is",round(regr.eval(train_data[,"inc"], pred5_train$pred)[4],4)*100,"%")

#Test
cat("Error metrics on test data for k=5")
regr.eval(test_data[,"inc"], pred5_test$pred)
cat("MAPE for k=5 is",round(regr.eval(test_data[,"inc"], pred5_test$pred)[4],4)*100,"%")

# .........k = 7........................#
#Predicting on train data

pred7_train = knn.reg(train = train_data[,ind_variables],test = train_data[,ind_variables],
                      y = train_data$inc, k = 7)

pred7_test = knn.reg(train = train_data[,ind_variables],test = test_data[,ind_variables],
                     y = train_data$inc, k = 7)
#Error metrics for regression
#Train
cat("Error metrics on train data for k=7")
regr.eval(train_data[,"inc"], pred7_train$pred)
cat("MAPE for k=7 is",round(regr.eval(train_data[,"inc"], pred7_train$pred)[4],4)*100,"%")

#Test
cat("Error metrics on test data for k=7")
regr.eval(test_data[,"inc"], pred7_test$pred)
cat("MAPE for k=7 is",round(regr.eval(test_data[,"inc"], pred7_test$pred)[4],4)*100,"%")

# Testing the final model performance on evaluation data 
pred_eval = knn.reg(train = train_data[,ind_variables],test = eval_data[,ind_variables],
                    y = train_data$inc, k = 5)

cat("Error metrics on eval data for k=5")
regr.eval(eval_data[,"inc"], pred_eval$pred)
cat("MAPE for k=5 is",round(regr.eval(eval_data[,"inc"], pred_eval$pred)[4],4)*100,"%")


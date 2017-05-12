rm(list=ls(all=TRUE))

setwd("C:/Users/jeevan/Desktop/KNN")

# Load required libraries
library(vegan)
library(dummies)
library(FNN)
library(DMwR)
library(class)

attr = c('id', 'age', 'exp', 'inc', 'zip', 'family', 
         'ccavg', 'edu', 'mortgage', 'loan', 
         'securities', 'cd', 'online', 'cc')



# Read the data using csv file
data = read.csv(file = "UniversalBank.csv", 
                header = TRUE, col.names = attr)
str(data)
# Removing the id, zip and experience. 
drop_Attr = c("id", "zip", "exp")
attr = setdiff(attr, drop_Attr)
data = data[, attr]
rm(drop_Attr)

# Convert attribute to appropriate type  
cat_Attr = c("family", "edu", "securities", 
             "cd", "online", "cc", "loan")

num_Attr = setdiff(attr, cat_Attr)

rm(attr)

cat_Data <- data.frame(sapply(data[,cat_Attr], as.factor))
num_Data <- data.frame(sapply(data[,num_Attr], as.numeric))

data = cbind(num_Data, cat_Data)
rm(cat_Data, num_Data)

# Do the summary statistics and check for missing values and outliers.
summary(data)

#------------------------------------------------------


# Build the classification model.
reg_Ind_Num_Attr = num_Attr
reg_Ind_Cat_Attr = setdiff(cat_Attr, "loan")

# Standardizing the numeric data
reg_Data = decostand(data[,reg_Ind_Num_Attr], "range") 

#reg_Data1 = decostand(data[,reg_Ind_Num_Attr],"total")
rm(reg_Ind_Num_Attr)

str(data)
# Convert all categorical attributes to numeric 
# 1. Using dummy function, convert education and family categorical attributes into numeric attributes 
edu = dummy(data$edu)
family = dummy(data$family)
reg_Data = cbind(reg_Data, edu, family)
reg_Ind_Cat_Attr = setdiff(reg_Ind_Cat_Attr, c("edu", "family"))
#str(reg_Data)
rm(edu, family)

# 2. Using as.numeric function, convert remaining categorical attributes into numeric attributes 
reg_Data = cbind(reg_Data, sapply(data[,reg_Ind_Cat_Attr], as.numeric))
rm(reg_Ind_Cat_Attr)
reg_Ind_Attr = names(reg_Data)

# Append the Target attribute 
reg_Data = cbind(reg_Data, loan=data[,"loan"]) 

str(reg_Data)
summary(reg_Data)

# Divide the data into test, train and eval
set.seed(123)

rowIDs = 1:nrow(reg_Data)
train_RowIDs =  sample(rowIDs, length(rowIDs)*0.6)
test_RowIDs = sample(setdiff(rowIDs, train_RowIDs), length(rowIDs)*0.2)
eval_RowIDs = setdiff(rowIDs, c(train_RowIDs, test_RowIDs))
rm(rowIDs)

train_Data = reg_Data[train_RowIDs,]
test_Data = reg_Data[test_RowIDs,]
eval_Data = reg_Data[eval_RowIDs,]
rm(train_RowIDs, test_RowIDs, eval_RowIDs)
 
# Check how records are split with respect to target attribute.
table(reg_Data$loan)
table(train_Data$loan)
table(test_Data$loan)
table(eval_Data$loan)
rm(reg_Data)

# Build best KNN model 

# k = 1
pred_Train = knn(train_Data[,reg_Ind_Attr], 
                 train_Data[,reg_Ind_Attr], 
                 train_Data$loan, k = 1)

cm_Train = table(pred_Train, train_Data$loan)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

pred_Test = knn(train_Data[,reg_Ind_Attr], 
                test_Data[,reg_Ind_Attr], 
                train_Data$loan, k = 1)

cm_Test = table(pred_Test, test_Data$loan)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
rm(pred_Test, cm_Test)

accu_Train
accu_Test

# k = 3 
pred_Train = knn(train_Data[,reg_Ind_Attr], 
                 train_Data[,reg_Ind_Attr], 
                 train_Data$loan, k = 3)


cm_Train = table(pred_Train, train_Data$loan)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

pred_Test = knn(train_Data[,reg_Ind_Attr], 
                test_Data[,reg_Ind_Attr], 
                train_Data$loan, k = 3)

cm_Test = table(pred_Test, test_Data$loan)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
rm(pred_Test, cm_Test)

accu_Train
accu_Test

# k = 5
pred_Train = knn(train_Data[,reg_Ind_Attr], 
train_Data[,reg_Ind_Attr], 
train_Data$loan, k = 5)

cm_Train = table(pred_Train, train_Data$loan)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

pred_Test = knn(train_Data[,reg_Ind_Attr], 
                test_Data[,reg_Ind_Attr], 
                train_Data$loan, k = 5)

cm_Test = table(pred_Test, test_Data$loan)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
rm(pred_Test, cm_Test)

accu_Train
accu_Test

# k = 7
pred_Train = knn(train_Data[,reg_Ind_Attr], 
                 train_Data[,reg_Ind_Attr], 
                 train_Data$loan, k = 7)

cm_Train = table(pred_Train, train_Data$loan)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

pred_Test = knn(train_Data[,reg_Ind_Attr], 
                test_Data[,reg_Ind_Attr], 
                train_Data$loan, k = 7)

cm_Test = table(pred_Test, test_Data$loan)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
rm(pred_Test, cm_Test)

accu_Train
accu_Test

rm(accu_Test, accu_Train)

# Condensing train Data
keep = condense(train_Data, train_Data$loan)
length(keep)
nrow(train_Data)
keep

pred = knn(train_Data[keep,reg_Ind_Attr], 
           test_Data[,reg_Ind_Attr], 
           train_Data$loan[keep], k=5)
rm(keep)
cm <- table(pred, test_Data$loan)
cm
accu=sum(diag(cm))/sum(cm)
accu


# Test the final model performance on evaluation data and report the results
pred = knn(train_Data[keep,reg_Ind_Attr], 
           eval_Data[,reg_Ind_Attr], 
           train_Data$loan[keep], k=5)
cm <- table(pred, eval_Data$loan)
cm
accu=sum(diag(cm))/sum(cm)
accu

rm(pred, cm, accu)
rm(eval_Data, test_Data, train_Data, reg_Ind_Attr)

#--------------------------------------------------------

# Build the Regression model. 

clas_Ind_Num_Attr = setdiff(num_Attr, "inc")
clas_Ind_Cat_Attr = cat_Attr

# Standardizing the numeric data
clas_Data = decostand(data[,clas_Ind_Num_Attr], "range") 
rm(clas_Ind_Num_Attr)

# Convert all categorical attributes to numeric 
# 1. Using dummy function, convert education and family categorical attributes into numeric attributes 

edu = dummy(data$edu)
family = dummy(data$family)
clas_Data = cbind(clas_Data, edu, family)
clas_Ind_Cat_Attr = setdiff(clas_Ind_Cat_Attr, c("edu", "family"))
rm(edu, family)

# 2. Using as.numeric function, convert remaining categorical attributes into numeric attributes 
clas_Data = cbind(clas_Data, sapply(data[,clas_Ind_Cat_Attr], as.numeric))
rm(clas_Ind_Cat_Attr)
clas_Ind_Attr = names(clas_Data)

# Append the Target attribute 
clas_Data = cbind(clas_Data, inc=data[,"inc"]) 

str(clas_Data)
summary(clas_Data)

# Divide the data into test, train and eval
set.seed(12345)

rowIDs = 1:nrow(clas_Data)
train_RowIDs =  sample(rowIDs, length(rowIDs)*0.6)
test_RowIDs = sample(setdiff(rowIDs, train_RowIDs), length(rowIDs)*0.2)
eval_RowIDs = setdiff(rowIDs, c(train_RowIDs, test_RowIDs))
rm(rowIDs)

train_Data = clas_Data[train_RowIDs,]
test_Data = clas_Data[test_RowIDs,]
eval_Data = clas_Data[eval_RowIDs,]
rm(train_RowIDs, test_RowIDs, eval_RowIDs)

# Check how records are split with respect to target attribute.
summary(clas_Data$inc)
summary(train_Data$inc)
summary(test_Data$inc)
summary(eval_Data$inc)
rm(clas_Data)

# Build best KNN model 

# k = 1
pred_Train = knn.reg(train = train_Data[,clas_Ind_Attr], 
                     test = train_Data[,clas_Ind_Attr],
                     y = train_Data$inc, k = 1)

pred_Test = knn.reg(train = train_Data[,clas_Ind_Attr], 
                    test = test_Data[,clas_Ind_Attr],
                    y = train_Data$inc, k = 1)

regr.eval(train_Data[,"inc"], pred_Train$pred)
regr.eval(test_Data[,"inc"], pred_Test$pred)

rm(pred_Train, pred_Test)

# k = 3 
pred_Train = knn.reg(train = train_Data[,clas_Ind_Attr], 
                     test = train_Data[,clas_Ind_Attr],
                     y = train_Data$inc, k = 3)

pred_Test = knn.reg(train = train_Data[,clas_Ind_Attr], 
                    test = test_Data[,clas_Ind_Attr],
                    y = train_Data$inc, k = 3)

regr.eval(train_Data[,"inc"], pred_Train$pred)
regr.eval(test_Data[,"inc"], pred_Test$pred)

rm(pred_Train, pred_Test)

# k = 5

pred_Train = knn.reg(train = train_Data[,clas_Ind_Attr], 
                     test = train_Data[,clas_Ind_Attr],
                     y = train_Data$inc, k = 5)

pred_Test = knn.reg(train = train_Data[,clas_Ind_Attr], 
                    test = test_Data[,clas_Ind_Attr],
                    y = train_Data$inc, k = 5)

regr.eval(train_Data[,"inc"], pred_Train$pred)
regr.eval(test_Data[,"inc"], pred_Test$pred)

rm(pred_Train, pred_Test)

# k = 7
pred_Train = knn.reg(train = train_Data[,clas_Ind_Attr], 
                     test = train_Data[,clas_Ind_Attr],
                     y = train_Data$inc, k = 7)

pred_Test = knn.reg(train = train_Data[,clas_Ind_Attr], 
                    test = test_Data[,clas_Ind_Attr],
                    y = train_Data$inc, k = 7)

regr.eval(train_Data[,"inc"], pred_Train$pred)
regr.eval(test_Data[,"inc"], pred_Test$pred)

rm(pred_Train, pred_Test)

# Test the final model performance on evaluation data and report the results
pred_Eval = knn.reg(train = train_Data[,clas_Ind_Attr], 
                    test = eval_Data[,clas_Ind_Attr],
                    y = train_Data$inc, k = 5)

regr.eval(eval_Data[,"inc"], pred_Eval$pred)

rm(clas_Ind_Attr, pred_Eval)

rm(eval_Data, test_Data, train_Data)
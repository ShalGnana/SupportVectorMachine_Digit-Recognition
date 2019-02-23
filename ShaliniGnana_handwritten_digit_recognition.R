################Handwritten digit recognition using SVM########################
#1. Business Understanding
#2. Data Understanding
#3. Data Preparation
#4. Model Building
#5. Cross validation
######################################################################################

#1. Business Understanding
# To develop a model using Support Vector Machine which should correctly classify the handwritten digits
#based on the pixel values given as features.
###################################################################################

#2. Data understanding 
#MNIST data is a large database of handwritten digits where we have pixel
#values of each digit along with its label. Sample the data and build the model which would make the computation faster.
#take 15% of train and test data seperately


######################################################################################################

#3. Data Preparation
# intsall library and load the required libraries 
install.packages("MASS")
install.packages("kernlab")
install.packages("dplyr")
install.packages("readr")
install.packages ("ggplot2")
install.packages("gridExtra")
install.packages("caTools")
install.packages("gbm")
install.packages("lattice")
install.packages("e1071")
install.packages("caret",dependencies=TRUE)
install.packages("RcppRoll")
install.packages("ddalpha")
install.packages("DEoptimR")
install.packages("dimRed")
install.packages("gower")
library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)
library(caTools)
library(lattice)
library(MASS)
library(gbm)
library(e1071)

#set working directory 

mnist_train<-read.csv("mnist_train.csv", stringsAsFactors = F, header= F)
mnist_test<-read.csv("mnist_test.csv", stringsAsFactors = F, header=F)

View(mnist_train) #60000 obs. of 785 variables, no column names
View(mnist_test) #10000 obs. of 785 variables, no column names

#set column names for 1st column as that represents the digits ie labels

colnames(mnist_train)[1]<-"label"
colnames(mnist_test)[1]<-"label"

View(mnist_train)
View(mnist_test)

#Data cleaning and preparation

#understanding structure of datasets
str(mnist_train)
str(mnist_test) 
#label column is of integer type in both train and test datasets. Convert it to factor type.

mnist_train$label<-as.factor(mnist_train$label)
mnist_test$label<-as.factor(mnist_test$label)

summary(mnist_train)
summary(mnist_test)
str(mnist_train)
str(mnist_test)

#checking missing values
sapply(mnist_train, function(x) sum(is.na(x))) #no NA values
sapply(mnist_test, function(x) sum(is.na(x))) #no NA values

#checking duplicate rows
sum(duplicated(mnist_train)) #0 indicates no duplicate rows
sum(duplicated(mnist_test)) #0 indicates no duplicate rows

#Taking 10% sample of train and test dataset as computation time will be too much for main dataset


set.seed(100)

train_indices<-sample(1:nrow(mnist_train), 6000) #10% sample of train dataset is 9000 rows

train<-mnist_train[train_indices, ]

summary(train) #pixel value goes up to 255
View(train) #some columns have only zeroes.Scaling is needed

test_indices<-sample(1:nrow(mnist_test), 1000) #15% sample of test dataset is 1500
test<-mnist_test[test_indices, ]

summary(test)
View(test)
#Scaling the data

#Taking the max value of pixel 255 to scale data

train[, 2:ncol(train)]<-train[, 2:ncol(train)]/255

test<-cbind(label=test[ ,1], test[ ,2:ncol(test)]/255)

View(train)
View(test)

#4. Model building 

#4.1 Linear model-SVM at Cost(C)=1

model1<- ksvm(label~., data=train, scaled= FALSE, kernel="vanilladot", C=1) #setting default kernel parameters

print(model1) #number of support vectors:1978; training error: 5e-04

#Predicting the model results

evaluate1<-predict(model1, test)

#Confusion Matrix: Finding accuracy, specificity and sensitivity

confusionMatrix(evaluate1, test$label)

#Accuracy: 92%
#Specificity across classes > 99%
#Sensitivity across classes > 83%

#4.2 Linear model -SVM at Cost C=10

model2<-ksvm(label~., data=train, scaled=FALSE, kernel ="vanilladot", C=10)
print(model2) #number of support vectors: 1975; training error: 0

#Predicting the model results
evaluate2<-predict(model2, test)

#Confusion Matrix

confusionMatrix(evaluate2, test$label)

# Accuracy: 91%
#Specificity across classes >99%
#Sensitivity across classes >82%

#With C=10, model performance has come down. It may be overfitting.

#Using cross validation to optimise C


trainControl<- trainControl(method="cv", number=3) #3-fold cross validation
metric<-"Accuracy"

#Making a grid of C values


grid<- expand.grid(C=c(0.01, 0.1, 1,10, 100))

# Performing 3-fold cross validation
fit.svm <- train(label~., data=train, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

#Print and plot
print(fit.svm)
plot(fit.svm)

#C=0.1 is the best value of C showing 92% accuracy

##########################################################################
#Validating the model after cross validation on test data

evaluate_linear_test<-predict(fit.svm, test)
confusionMatrix(evaluate_linear_test, test$label)

#Accuracy: 92.5% has improved 
#Specificity >98% 
#Sensitivity >88% has improved 
# This indicates that low value of C=0.1 has made the model generic.

################################################################################

# Checking Polynomial Kernel model

# Using C=1

model1_poly <- ksvm(label~., data=train, scaled = FALSE, kernel="polydot", C=1)

print(model1_poly) # Training error: 5e-04, degree=1, scale=1, offset =1

#Predicting the model results

evaluate1_poly<-predict(model1_poly, test)

#Confusion Matrix: Finding accuracy, specificity and sensitivity

confusionMatrix(evaluate1_poly, test$label)

#Accuracy: 92%
#Specificity across classes >98%
#Sensitivity across classes >83%

#Using C=5

model2_poly<-ksvm(label~., data=train, scaled=FALSE, kernel="polydot", C=5)
print(model2_poly)

evaluate2_poly<-predict(model2_poly, test)

confusionMatrix(evaluate2_poly, test$label)

#Accuracy : 91%
#Specificity across classes > 98%
#Sensitivity across classes > 81%

#There is a drop in accuracy for C=5.

# Check cross validation C values between 1 to 5 to find optimised C

#Using 2-fold cross validation to save run time

trainControl<-trainControl(method="cv", number=2)

metric <-"Accuracy"

#Use parallel processing to save computational time

install.packages("doParallel")
library(doParallel)
detectCores(logical=FALSE) #gives core=2
cl<-makePSOCKcluster(2)
registerDoParallel(cl)


grid_poly<-expand.grid(C=c(0.01, 0.1, 1,10, 100), degree= c(1,2,3,4,5), scale=c(10,20,30,40,50))

#try different scale grid_poly<-expand.grid( C=c(0.01, 0.1, 1, 10, 100), degree=c(1,2,3,4,5), scale=c(1,2,3,4,5))

fit.poly<- train(label~., data=train, method="svmPoly", metric=metric, tuneGrid=grid_poly, trControl=trainControl)

stopCluster(cl)
print(fit.poly)#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were Accuracy: 94% degree = 2, scale = 20 and C = 0.01.

plot(fit.poly)


#Validating the model after cross validation on test data

evaluate_poly_test<-predict(fit.poly, test)
confusionMatrix(evaluate_poly_test, test$label)

#Accuracy: 95.3%
#Specificity across classes > 99%
#Sensitivity across classes >89%

# This is achieved for C=0.01, degree =2, scale=20 for 2 fold cross validation

#Applying these C, scale and degree optimised values to the model

model3_poly<-ksvm(label~., data=train, scaled=FALSE, kernel="polydot", C=0.01)
#Number of support vectors: 2656, training error: 0.05333

print(model3_poly)

evaluate3_poly<-predict(model3_poly, test)

confusionMatrix(evaluate3_poly, test$label)
#Accuracy: 93.3%
#specificity across classes >99%
#Sensitivity across classes >89%
###################################################################################


#####################Radial Kernel#################################################
#Using C=1 and sigma=0.01 (low value of sigma chosen to prevent overfitting)

Model1_rbf <- ksvm(label~ ., data = train, scale = FALSE, kernel = "rbfdot", C=1)
Eval1_rbf<- predict(Model1_rbf, test)
confusionMatrix(Eval1_rbf,test$label)
#Accuracy: 95.5%
#Specificity across classes is >=99%
#Sensitivity across classes is >90%

#######################Hyperparameter tuning and cross validation############


trainControl <- trainControl(method="cv", number=2)#Using 2 fold to save computational time.
metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
detectCores(logical=FALSE) #gives core=2
cl<-makePSOCKcluster(2)
registerDoParallel(cl)
set.seed(7)
grid <- expand.grid(.sigma=c(0.01, 0.025, 0.05), .C=c(0.01, 0.1, 1, 5, 10) )

fit.rbf <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl, allowparallel=TRUE)

stopCluster(cl)

print(fit.rbf)#Accuracy: 96%, the best value of sigma = 0.025 and C = 5.
#Accuracy drops at C>5 and higher value of sigma >0.25 is overfitting.
plot(fit.rbf)

#Validating the model after cross validation on test data

evaluate_rbf_test<-predict(fit.rbf, test)
confusionMatrix(evaluate_rbf_test, test$label)
#Accuracy: 96%
#Specificity across classes > 99%
#Sensitivity across classes > 91%
############################################################################

#RBF Kernel model has shown the best accuracy at 96% compared to linear and polynomial kernel.
#Low values of sigma were tried to reduce overfitting
#2-fold cross validation and 10% sample used across train and test datasets 
#to save computational time.

#RBF Kernel model at sigma=0.25 and C=5 gives best results:

#Accuracy: 96%
#Specificity across classes > 99%
#Sensitivity across classes > 91%

###################################################################################

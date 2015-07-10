#######################################
# Prepare for Classification Modeling #
#######################################

# Set Working Directory
setwd('C:/Users/mcohen/Documents/R/BlackRock')

# Clear Workspace #
rm(list=ls())

# Set seed
set.seed(345)

# Read in Data #
data = read.delim('har_dataset.txt',header=TRUE,sep=";")

# simplify some variable names #
names(data)[1]='user'
names(data)[4]='height_m'
names(data)[6]='bmi'
data$z4 = as.numeric(data$z4)

# Assign values to learning (training & validation) and test data frames by user stratified sampling
users = levels(data$user)
learn = data.frame(matrix(ncol = 1e4,nrow = 0))
names(learn) = names(data)
test = learn
for (ii in 1:length(users)){
  userdata = data[data$user==users[ii],]
  ind = sample(1:nrow(userdata),size=nrow(userdata)*0.9);
  learn = rbind(learn,userdata[ind,])
  test = rbind(test,userdata[-ind,])
}

summary(learn)
summary(test)

#######################
# Classification Tree #
#######################

library(rpart)

# grow tree 
fit_tree = rpart(class ~ user + x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 + z1 + z2 + z3 + z4,data = learn)
plot(fit_tree, uniform = TRUE)
text(fit_tree,use.n = TRUE, cex = 0.75)
summary(fit_tree)
plotcp(fit_tree)
printcp(fit_tree)

# Prune Tree
fit_tree2 = prune(fit_tree, cp = 0.01)
plot(fit_tree2, uniform = TRUE)
text(fit_tree2,use.n = TRUE, cex = 0.75)
summary(fit_tree2)

# Learn Sample Fit
printcp(fit_tree2)

# Test Sample Validation
tree_learn_fit = table(predict(fit_tree2,learn,type="class"),learn$class)
tree_learn_error = 1-sum(diag(tree_learn_fit))/sum(tree_learn_fit)

tree_learn_error

#########################
# Classification Forest #
#########################

library(randomForest)
library(foreach)

rf_model = foreach(ntree = rep(100,4), .combine = combine, .packages='randomForest') %dopar% {
  randomForest(class ~ gender + user + x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 + z1 + z2 + z3 + z4,data=learn,ntree=ntree)
}

# Learn Sample Fit
rf_pred = table(predict(rf_model,newdata=learn,type="class"),learn$class)
rf_error = 1-sum(diag(rf_pred))/sum(rf_pred)
rf_error

# Test sample validation
rf_pred_test<-table(predict(rf_model,newdata=test,type="class"),test$class)
rf_error_test = 1-sum(diag(rf_pred_test))/sum(rf_pred_test)
rf_error_test

#######
# svm #
#######

library(e1071)

svm_model = svm(class ~ gender + user + x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 + z1 + z2 + z3 + z4,data = learn)

# Learn Sample Fit
svm_pred = table(predict(svm_model,newdata=learn,type="class"),learn$class)
svm_error = 1-sum(diag(svm_pred))/sum(svm_pred)
svm_error

# Test Sample Validation
svm_pred_test<-table(predict(svm_model,newdata=test,type="class"),test$class)
svm_error_test = 1-sum(diag(svm_pred_test))/sum(svm_pred_test)
svm_error_test

cat('The random forest classifier appears to classify best for this test sample with a misclassification rate of approximately', ceiling(1000*rf_error_test)/10,'%', fill = TRUE)

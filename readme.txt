Files contained in this repository:
"har_dataset.txt" - a semicolon delimited flat data file of human activity recognition for from http://groupware.les.inf.puc-rio.br/har#dataset
"main.R" - and R script to load data, create learn and test sample, and three modeling algrithms to for multinomial classifcation, which include, a classification tree, a random classification forest, and a support vector machine

This program considers three algorithms to classify human activity: "sitting" "sittingdown" "standing"   "standingup" "walking"  based upon twelve features captured by wearable biometric sensors, the data also identifies four subjects, 2 female, and 2 male. 

The R script requires the following packages with dependencies: rpart, randomForest, foreach, and e1071

The observed Regression Tree classifier accuracy is on the order of 99.6-99.7% for a 10 percent holdout sample. 
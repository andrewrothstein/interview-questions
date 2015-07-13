#Download and unzip file from url
temp <- tempfile()
download.file("http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip", temp)
unzipdat <- unz(temp, "dataset-har-PUC-Rio-ugulino.csv")
#Read data from file into variable
data<-read.csv(unzipdat, head=TRUE, sep = ";")
unlink(temp)

#Omit missing data
data<-na.omit(data)
#Using caret library
library(caret)
#create partiton for training/testing with 90% data
partition <- createDataPartition(y=data$class,p=0.9,list=FALSE)
trainData <- data[partition,]
testData <- data[-partition,]
#list of top algorithms for this data set
algolist<- c("C5.0", "nb")
#initialize dataframe to output results
compareTable <- data.frame(Algorithm=character(), Accuracy=numeric(), Duration=numeric(), stringsAsFactors=FALSE)

#initialize x,y values for train method
x = trainData
y = trainData$class

#iterate through algolist
for (i in 1:length(algolist)) {
  algo <- algolist[i]
  print(paste("Algorithm = ", algo ))
  #set start time to calculate duration
  startTime <- as.integer(Sys.time())
  #train using cv
  model = train(x,y,algo,trControl=trainControl(method='cv',number=10))
  #predict data using model
  results <- predict(model, testData)
  #generate confuson matrix using predicted results and 10% test data
  confmatrix<- confusionMatrix(results, testData$class)
  endTime <- as.integer(Sys.time())
  
  compareTable[i,1] <- algo
  compareTable[i,2] <- round(as.numeric(confmatrix$overall[1]) * 100, 2) 
  compareTable[i,3] <- ((endTime-startTime)/60)

}

print(compareTable)

#C5.0 performed with 100% accuracy and executed quicker, so this is our best classifier
model = train(x,y,"C5.0",trControl=trainControl(method='cv',number=10))
print(model)
#uncomment to view warnings post execution
#warnings()

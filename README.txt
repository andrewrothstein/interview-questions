Instructions for testing: 

Please load “classifier.R” into a development environment. I used RStudio to write this. Once loaded, please select all and run. 

Strategy and Execution:

In order to decide which method to use, I had a look at the caret library algorithms and measured performance for the data set. The decision tree algorithm, C5.0, and Naive Bayes, nb, consistently performed with the highest accuracy and quickest times (in comparison to bagFDA, nnet, rf). These two algorithms are shown in the R script, with performance comparisons printed. Ultimately, the decision tree strategy outperformed Naive Bayes with 100% accuracy and a quicker execution time of just over 7 and a half minutes. Thus, this was the algorithm of choice for this dataset.

Overview of code: 

The zip file is first loaded from the url given and the csv data is unzipped into the data variable.

I then load the caret library, which conveniently contains a createDataPartition function. Using this function, I can specify data I used for training, as well as the out-of-sample test data. As per the instructions, this validation data is 10%. 

I then iterate through the list of algorithms to generate a model using the train function, track the start and end times to calculate duration, and record the results into the comparison table. Using the cross-validated method for training with 10 fold resampling yielded 100% accuracy for the C5.0 algorithm, so I took that as a sufficient train control (using this train method with no train control actually resulted in just 29% accuracy).

Due to the 100% accurate C5.0 method, the only way nb could be a superior choice is if it executed at a faster time. The comparison table shows while nb was 98.9% accurate, it did not run as fast or match the accuracy of the C5.0 algorithm. 


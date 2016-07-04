# Machine Learning Final Project
Marc Hidalgo  
July 2, 2016  
##Background 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of this project is to predict the manner in which subjects have performed the exercise. This is the "classe" variable in the training set. 

##Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

###Load Necessary Packages/Read in the Data

Load the appropriate packages required and read in the data files from CSV. The code assume the files are already downloaded.


```r
setwd("/Users/Marc/coursera/")
library(ggplot2)
library(lattice)
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
```

```
## R session is headless; GTK+ not initialized.
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
## if necessary, download the data files
trainFile <- "pml-training.csv"
if (!file.exists(trainFile)){
  fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(fileURL, trainFile, method="curl")
}
testFile <- "pml-testing.csv"
if (!file.exists(testFile)){
  fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileURL, testFile, method="curl")
}
trainSet <- read.csv(trainFile)
testSet <- read.csv(testFile)
```

###Partition the Data

Split the training data into training and testing partitions (60% and 40%) and use the testSet as the validation sample. Use cross validation within the training partition to improve the model fit and then do an out-of-sample test with the testing partition. Set a seed first so that this part is reproducible.


```r
set.seed(100)
dataPart <- createDataPartition(y=trainSet$classe, p=0.6, list=FALSE)
trainingPart <- trainSet[dataPart, ]
testingPart <- trainSet[-dataPart, ]
```

###Remove Extraneous Data

Remove variables that are almost always NA since their impact will likely be minimal. Remove variables with near zero variance (i.e., practically constant). And remove variables that likely don't impact the results. We perform that analysis on the trainingPart data set, since it's the larger of the two, and apply the results to both trainingPart and testingPart.


```r
# remove generally NA variables
mostlyNA <- sapply(trainingPart, function(x) mean(is.na(x))) > 0.95
trainingPart <- trainingPart[, mostlyNA==F]
testingPart <- testingPart[, mostlyNA==F]

# remove variance near zero
nearZero <- nearZeroVar(trainingPart)
trainingPart <- trainingPart[, -nearZero]
testingPart <- testingPart[, -nearZero]

# remove variables that likely don't impact the results. These happen to be the first five variables
trainingPart <- trainingPart[, -(1:5)]
testingPart <- testingPart[, -(1:5)]
```

###Build the Models

Use three different method to build a model of the data:

1. Decision trees with CART (rpart)

2. Stochastic gradient boosting trees (gbm)

3. Random forest decision trees (rf)

###Decision Tree


```r
dtModel<-train(classe ~ ., data=trainingPart, method="rpart")
fancyRpartPlot(dtModel$finalModel,cex=.5,under.cex=1,shadow.offset=0)
```

![](machineLearningProject_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
predictDt<-predict(dtModel, testingPart)
dtMat<-confusionMatrix(predictDt, testingPart$classe)
dtMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1994  652  628  550  123
##          B   36  496   36  241  110
##          C  163  370  704  432  295
##          D    0    0    0    0    0
##          E   39    0    0   63  914
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5236          
##                  95% CI : (0.5125, 0.5347)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3787          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8934  0.32675  0.51462   0.0000   0.6338
## Specificity            0.6521  0.93315  0.80550   1.0000   0.9841
## Pos Pred Value         0.5052  0.53972  0.35845      NaN   0.8996
## Neg Pred Value         0.9390  0.85246  0.88711   0.8361   0.9227
## Prevalence             0.2845  0.19347  0.17436   0.1639   0.1838
## Detection Rate         0.2541  0.06322  0.08973   0.0000   0.1165
## Detection Prevalence   0.5031  0.11713  0.25032   0.0000   0.1295
## Balanced Accuracy      0.7727  0.62995  0.66006   0.5000   0.8090
```

```r
dtAccuracy <- dtMat$overall[[1]] 
```

###Stochastic Gradient Boosting Trees


```r
gbmModel<-train(classe ~.,data=trainingPart,method="gbm",verbose=FALSE,trControl = trainControl(number=5,repeats=1))
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```r
predictGbm<-predict(gbmModel,testingPart)
gbmMat<-confusionMatrix(predictGbm,testingPart$classe)
gbmMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231   12    0    1    2
##          B    1 1488    8    3    8
##          C    0   15 1354   13    2
##          D    0    3    6 1266   11
##          E    0    0    0    3 1419
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9888         
##                  95% CI : (0.9862, 0.991)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9858         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9802   0.9898   0.9844   0.9840
## Specificity            0.9973   0.9968   0.9954   0.9970   0.9995
## Pos Pred Value         0.9933   0.9867   0.9783   0.9844   0.9979
## Neg Pred Value         0.9998   0.9953   0.9978   0.9970   0.9964
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1897   0.1726   0.1614   0.1809
## Detection Prevalence   0.2863   0.1922   0.1764   0.1639   0.1812
## Balanced Accuracy      0.9984   0.9885   0.9926   0.9907   0.9918
```

```r
gbmAccuracy <- gbmMat$overall[[1]]
```
###Random Forest Decision Tree


```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
rfModel<-randomForest(classe ~., data=trainingPart)
predictRf<-predict(rfModel,testingPart)
rfMat<-confusionMatrix(predictRf,testingPart$classe)
rfMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    3    0    0    0
##          B    0 1512    6    0    0
##          C    0    3 1361    8    0
##          D    0    0    1 1278    4
##          E    0    0    0    0 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9968          
##                  95% CI : (0.9953, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.996           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9960   0.9949   0.9938   0.9972
## Specificity            0.9995   0.9991   0.9983   0.9992   1.0000
## Pos Pred Value         0.9987   0.9960   0.9920   0.9961   1.0000
## Neg Pred Value         1.0000   0.9991   0.9989   0.9988   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1927   0.1735   0.1629   0.1833
## Detection Prevalence   0.2849   0.1935   0.1749   0.1635   0.1833
## Balanced Accuracy      0.9997   0.9975   0.9966   0.9965   0.9986
```

```r
rfAccuracy<-rfMat$overall[[1]]
```

Based on the results above, the accuracy for each model is given below:

1. Decision trees with CART (rpart): 0.5235789

2. Stochastic gradient boosting trees (gbm): 0.9887841

3. Random forest decision trees (rf): 0.9968137

The model with the best fit is therefore the random forest model. We'll next evaluate the random forest model against the test dataset. First, we estimate the out of sample error.

###Out of Sample Error

Since we'll be using random forest model and since the accuracy of the random forest method is 0.9968137, the estimated out of sample error is given by:
100.00% - 99.6813663% = 0.3186337%.


###Data for Automated Quiz Evaluation

Use the test dataset and the random forest model to create the data to be submitted to the quiz.


```r
predictRfTest<-predict(rfModel,testSet)
predictRfTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

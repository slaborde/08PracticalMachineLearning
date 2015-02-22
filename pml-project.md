---
title: "Project PML"
date: "Friday, February 06, 2015"
output: html_document
---

##Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of this project is to predict the manner in which the participants did the exercise. This is the “classe” variable in the training set.

##Methodology

I will address the following steps:

1. Question
2. Input Data
3. Cross Validation
4. Model Specification
5. Parameters
6. Evaluation 

###Question

Six participants participated in a dumbell lifting exercise five different ways. The five ways, as described in the study, were “exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.”

By processing data gathered from accelerometers on the belt, forearm, arm, and dumbell of the participants in a machine learning algorithm, the question is can the appropriate activity quality (class A-E) be predicted ??

###Input Data

First load the training and test data provided trating empty values as NAs values. Then clean the data removing columns:

1. With NA values (columns with NAs values has more than 85% of NAs).
2. Non-numeric Variables
3. Near-Zero Values

The provided training set has 19.622 observations so we split the original training data in a new training set (70%) for perform cross validation and a test set (30%) put aside.


```
## Loading required package: lattice
## Loading required package: ggplot2
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```


```r
training <- read.csv("pml-training.csv" , na.strings=c("","NA"), header = TRUE)
testing <- read.csv("pml-testing.csv", na.strings=c("","NA"), header = TRUE)

#Remove NAs values
trainR <- training[,which(as.numeric(colSums(is.na(training))) == 0)]
tesintgR <- training[,which(as.numeric(colSums(is.na(testing))) == 0)]
#Remove Non-Numeric Variables
trainR <- trainR[,-(1:7)]
tesintgR <- tesintgR[,-(1:7)]

#Remove Near Zero Values
end <- ncol(trainR)
trainR[,-end] <- data.frame(sapply(trainR[,-end], as.numeric))
nzv <- nearZeroVar(trainR[, -end], saveMetrics = TRUE)
trainR <- trainR[,!as.logical(nzv$nzv)]

end <- ncol(tesintgR)
tesintgR[,-end] <- data.frame(sapply(tesintgR[,-end], as.numeric))
nzv <- nearZeroVar(tesintgR[, -end], saveMetrics = TRUE)
tesintgR <- tesintgR[,!as.logical(nzv$nzv)]

set.seed(33835)
inTrain <- createDataPartition(trainR$classe, p = 0.7, list = FALSE)
train <- trainR[inTrain,]
test <- trainR[-inTrain,]
```

##Cross Validation

In order to avoid overfitting and to reduce out of sample errors, TrainControl is used to perform 7-fold cross validation.


```r
tc <- trainControl(method = "cv", number = 4, verboseIter = FALSE , preProcOptions = "pca", allowParallel = TRUE)
```

####Model Specification

We built and compared two models using tree and random forest algorithms. 

** First Model: Decision Tree **

```r
#treeModel <- train(classe ~ ., data = train, method = "rpart", trControl = tc)
treeModel2 <- rpart(classe ~ ., data = train, method = "class")

#fancyRpartPlot(treeModel$finalModel)
fancyRpartPlot(treeModel2)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-1.png) 

```r
#predictionTree <- predict(treeModel, newdata = test)
predictionTree2 <- predict(treeModel2, newdata = test, type = "class")
```

** Second Model: Random Forest **


```r
randForestModel2 <- randomForest(classe ~ . , data = train, method = "class")

#randForestModel <- train(classe ~ ., data = train, method = "rf", trControl = tc, prox = TRUE, allowParallel=TRUE,
#                         ntree = 5)

predictionRandForest2 <- predict(randForestModel2, newdata = test, type = "class")
```

###Conclusion

Random Forest algorithm performed better than Decision Trees.
Accuracy for Random Forest model was 0.9905 (95% CI: (0.9877, 0.9928)) compared to 0.4929 (95% CI: (0.4801, 0.5058)) for Decision Tree model. 

The random Forest model is choosen. The accuracy of the model is 0.995. The expected out-of-sample error is estimated at 0.005, or 0.5%. The expected out-of-sample error is calculated as 1 - accuracy for predictions made against the cross-validation set.


```r
#confusionMatrix(predictionTree, test$classe)
confusionMatrix(predictionTree2, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1521  229   21   92   50
##          B   55  684   66   56   91
##          C   49  117  871  157  138
##          D   19   74   58  618   54
##          E   30   35   10   41  749
## 
## Overall Statistics
##                                           
##                Accuracy : 0.755           
##                  95% CI : (0.7438, 0.7659)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6886          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9086   0.6005   0.8489   0.6411   0.6922
## Specificity            0.9069   0.9435   0.9051   0.9583   0.9758
## Pos Pred Value         0.7951   0.7185   0.6539   0.7509   0.8659
## Neg Pred Value         0.9615   0.9078   0.9660   0.9316   0.9337
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2585   0.1162   0.1480   0.1050   0.1273
## Detection Prevalence   0.3251   0.1618   0.2263   0.1398   0.1470
## Balanced Accuracy      0.9078   0.7720   0.8770   0.7997   0.8340
```

```r
confusionMatrix(predictionRandForest2, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    1    0    0    0
##          B    2 1136    7    0    0
##          C    0    2 1019    7    2
##          D    0    0    0  957    4
##          E    0    0    0    0 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9958          
##                  95% CI : (0.9937, 0.9972)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9946          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9974   0.9932   0.9927   0.9945
## Specificity            0.9998   0.9981   0.9977   0.9992   1.0000
## Pos Pred Value         0.9994   0.9921   0.9893   0.9958   1.0000
## Neg Pred Value         0.9995   0.9994   0.9986   0.9986   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1930   0.1732   0.1626   0.1828
## Detection Prevalence   0.2843   0.1946   0.1750   0.1633   0.1828
## Balanced Accuracy      0.9993   0.9977   0.9955   0.9960   0.9972
```


###Evaluation

Our Test data set comprises 20 cases. With an accuracy above 99% on our cross-validation data, we can expect that very few, or none, of the test samples will be missclassified.

** Making predictions with selected model **



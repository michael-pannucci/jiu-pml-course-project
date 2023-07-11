---
title: "JHU Practical Machine Learning Course Project"
author: "Michael Pannucci"
date: "2023-07-11"
output:   
  html_document:
    keep_md: true
---



## Executive Summary

We have been tasked with using accelerometer data to predict the outcome variable *classe*, which is a rating of how well someone performed a weightlifting movement using sensors based on the belt, forearm, arm, and dumbbell of six study participants. We have been provided a training set of nearly 20,000 observations and a testing set of 20 observations. We will apply a few machine learning algorithms learned in this course to model the training data and apply the one we think fits best to the testing data.

## Preliminaries

We begin by activating the needed libraries and setting a seed number for reproducibility.


```r
library(caret)
library(rattle)
set.seed(467)
```

## Loading the Data

Next, we will load in the data as linked from the course website.


```r
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainURL))
testing <- read.csv(url(testURL))
```

## Cleaning the Data

We will apply three cleaning steps to get the data ready for modeling.

1. Remove the first column (*X*) since it is just a index variable.
2. Remove columns with too many NA values (we use 75% as the cutoff here).
3. Remove columns with near-zero variance using the handy **nearZeroVariance()** function.


```r
training <- training[, colMeans(is.na(training)) < 0.75]
nzv <- nearZeroVar(training)
training <- training[, -c(1, nzv)]
```

## Training and Validation Sets

We will leave the provided testing set alone and use it later to, unsurprisingly, test our chosen model to predict the outcome of interest. Using the now-cleaned training data, let's split that into an active training set and a validation set.


```r
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
mytrain <- training[inTrain, ]
myval <- training[-inTrain, ]
```

## Cross-Validation

As it is good practice to cross-validate our training models, let's declare this using the **trainControl()** function using three folds.


```r
control <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
```

## Training Models

### Model 1: Decision Tree

Our first model will be a decision tree. Let's fit the model, use it to predict values from our validation set, and then save the confusion matrix.


```r
model1 <- train(classe ~ ., data = mytrain, method = "rpart", trControl = control)
fancyRpartPlot(model1$finalModel)
```

![](course_project_files/figure-html/model1-1.png)<!-- -->

```r
pred1 <- predict(model1, myval)
cfmatrix1 <- confusionMatrix(pred1, factor(myval$classe))
acc1 <- cfmatrix1$overall[1]
cfmatrix1
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1272  111    2    0    0
##          B  298  773  221  469  374
##          C  101  255  803  495  214
##          D    0    0    0    0    0
##          E    3    0    0    0  494
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5679          
##                  95% CI : (0.5551, 0.5806)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4544          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7599   0.6787   0.7827   0.0000  0.45656
## Specificity            0.9732   0.7130   0.7808   1.0000  0.99938
## Pos Pred Value         0.9184   0.3621   0.4299      NaN  0.99396
## Neg Pred Value         0.9107   0.9024   0.9445   0.8362  0.89087
## Prevalence             0.2845   0.1935   0.1743   0.1638  0.18386
## Detection Rate         0.2161   0.1314   0.1364   0.0000  0.08394
## Detection Prevalence   0.2353   0.3628   0.3174   0.0000  0.08445
## Balanced Accuracy      0.8665   0.6958   0.7817   0.5000  0.72797
```

### Model 2: Random Forest

Let's apply the same strategy for a model using a random forest.


```r
model2 <- train(classe ~ ., data = mytrain, method = "rf", trControl = control)
pred2 <- predict(model2, myval)
cfmatrix2 <- confusionMatrix(pred2, factor(myval$classe))
acc2 <- cfmatrix2$overall[1]
cfmatrix2
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    3    0    0
##          C    0    0 1023    2    0
##          D    0    0    0  960    0
##          E    0    0    0    2 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9988          
##                  95% CI : (0.9976, 0.9995)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9971   0.9959   1.0000
## Specificity            1.0000   0.9994   0.9996   1.0000   0.9996
## Pos Pred Value         1.0000   0.9974   0.9980   1.0000   0.9982
## Neg Pred Value         1.0000   1.0000   0.9994   0.9992   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1738   0.1631   0.1839
## Detection Prevalence   0.2845   0.1941   0.1742   0.1631   0.1842
## Balanced Accuracy      1.0000   0.9997   0.9983   0.9979   0.9998
```

### Model 3: Boosting

And finally, let's also perform a model using boosting.


```r
model3 <- train(classe ~ ., data = mytrain, method = "gbm", trControl = control, 
                verbose = FALSE)
pred3 <- predict(model3, myval)
cfmatrix3 <- confusionMatrix(pred3, factor(myval$classe))
acc3 <- cfmatrix3$overall[1]
cfmatrix3
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    6    0    0    0
##          B    0 1131    2    0    0
##          C    0    2 1022    1    0
##          D    0    0    2  960    4
##          E    0    0    0    3 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9966          
##                  95% CI : (0.9948, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9957          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9930   0.9961   0.9959   0.9963
## Specificity            0.9986   0.9996   0.9994   0.9988   0.9994
## Pos Pred Value         0.9964   0.9982   0.9971   0.9938   0.9972
## Neg Pred Value         1.0000   0.9983   0.9992   0.9992   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1922   0.1737   0.1631   0.1832
## Detection Prevalence   0.2855   0.1925   0.1742   0.1641   0.1837
## Balanced Accuracy      0.9993   0.9963   0.9977   0.9973   0.9978
```

### Summary

We see very high accuracy rates for the random forest and boosting models at over 99% each.



```r
summary <- rbind(acc1, acc2, acc3)
rownames(summary) <- c("Decision Tree", "Random Forest", "Boosting")
summary
```

```
##                Accuracy
## Decision Tree 0.5678845
## Random Forest 0.9988105
## Boosting      0.9966015
```

## Prediction on the Testing Set

Let's choose the random forest model and use it to predict the outcome variable on the testing set.


```r
testpred <- predict(model2, testing)
testpred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

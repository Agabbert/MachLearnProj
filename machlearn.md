# Machine Learning Course Project

The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict which activity they were performing. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

#### Load useful packages

```r
library(lubridate)
library(ggplot2)
library(tidyr)
library(dplyr)
library(knitr)
library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(gbm)
library(forecast)
library(e1071)
library(plyr)
library(randomForest)
library(MASS)
```

## Loading and preprocessing the data

```r
training.csv <- read.csv(file = "pml-training.csv")
dim(training.csv)
```

```
## [1] 19622   160
```

```r
valid.csv <- read.csv(file = "pml-testing.csv")
dim(valid.csv)
```

```
## [1]  20 160
```

```r
unique(training.csv$classe)
```

```
## [1] A B C D E
## Levels: A B C D E
```

```r
## Remove data not useful for prediction
training.csv <- training.csv[, -(1:7)]
training.csv <- tbl_df(training.csv)

training.csv <- bind_cols(lapply(training.csv[, 1:152], as.character), training.csv[,153])
training.csv <- bind_cols(lapply(training.csv[, 1:152], as.numeric), training.csv[,153])
training.csv <- training.csv[ , colSums(is.na(training.csv)) == 0]
training.csv[1:5, c(1:8, 53)]
```

```
## # A tibble: 5 × 9
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
##       <dbl>      <dbl>    <dbl>            <dbl>        <dbl>        <dbl>
## 1      1.41       8.07    -94.4                3         0.00         0.00
## 2      1.41       8.07    -94.4                3         0.02         0.00
## 3      1.42       8.07    -94.4                3         0.00         0.00
## 4      1.48       8.05    -94.4                3         0.02         0.00
## 5      1.48       8.07    -94.4                3         0.02         0.02
## # ... with 3 more variables: gyros_belt_z <dbl>, accel_belt_x <dbl>,
## #   classe <fctr>
```

```r
valid.csv <- valid.csv[, -(1:7)]
valid.csv <- tbl_df(valid.csv)

valid.csv <- bind_cols(lapply(valid.csv[, 1:152], as.character), valid.csv[,153])
valid.csv <- bind_cols(lapply(valid.csv[, 1:152], as.numeric), valid.csv[,153])
valid.csv <- valid.csv[ , colSums(is.na(valid.csv)) == 0]
valid.csv[1:5, c(1:8, 53)]
```

```
## # A tibble: 5 × 9
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
##       <dbl>      <dbl>    <dbl>            <dbl>        <dbl>        <dbl>
## 1    123.00      27.00    -4.75               20        -0.50        -0.02
## 2      1.02       4.87   -88.90                4        -0.06        -0.02
## 3      0.87       1.82   -88.50                5         0.05         0.02
## 4    125.00     -41.60   162.00               17         0.11         0.11
## 5      1.35       3.33   -88.60                3         0.03         0.02
## # ... with 3 more variables: gyros_belt_z <dbl>, accel_belt_x <dbl>,
## #   problem_id <int>
```

## Partition Data
The data set has been cleaned and is ready to be partitioned. The training data set was broken into a training and testing set of data and the "testing" provided set of data will be used for validation.

```r
set.seed(133) # for reproducibility
inTrain <- createDataPartition(training.csv$classe, p = .7)[[1]]
training <- training.csv[inTrain, ]
testing <- training.csv[-inTrain,]
```

## Modeling
Three models were created for comparison. Random forest was chosen since it has a reputation of being one of the most accurate prediction methods available. Linear discriminant analysis was chosen for the potentional for graphical representation and understanding. SVM was chosen as a method which lends itself to modeling an unknown distribution.

```r
mod.rf <- train(classe ~ ., data = training, method = "rf", ntree = 20)
mod.lda <- train(classe ~ ., data = training, method = "lda")
mod.svm <- svm(classe ~ ., data = training)
```

## Prediction and Out of Sample Error

```r
pred.rf <- predict(mod.rf, testing)
pred.lda <- predict(mod.lda, testing)
pred.svm <- predict(mod.svm, testing)

pred.df <- data.frame(pred.rf, pred.svm, classe = testing$classe)
head(pred.df)
```

```
##   pred.rf pred.svm classe
## 1       A        A      A
## 2       A        A      A
## 3       A        A      A
## 4       A        A      A
## 5       A        A      A
## 6       A        A      A
```

```r
# Random Forest
confusionMatrix(pred.rf, testing$classe)$overall[1]
```

```
##  Accuracy 
## 0.9904843
```

```r
# Linear Discriminant
confusionMatrix(pred.lda, testing$classe)$overall[1]
```

```
##  Accuracy 
## 0.7090909
```

```r
# SVM
confusionMatrix(pred.svm, testing$classe)$overall[1]
```

```
##  Accuracy 
## 0.9519116
```

Random forest showed the best accuracy, thus it was chosen for the model to use for prediction. In addition because the accuracy was so high it was assumed that combining the methods wouldn't yield significantly better results.

## Answer

```r
pred.valid <- predict(mod.rf, valid.csv); pred.valid
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
answer <- data.frame(pred.valid, problem.id = valid.csv$problem_id)
write.csv(answer, file = "prediction.csv", row.names = FALSE)
```

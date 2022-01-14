# Tidymodels Classification Example 
### Author:  Conrad Manske
### Date:  06/10/2021
### Last Edit:  01/13/2022  

## Introduction:
Well, I need a Classification example for my portfolio.  Why re-invent the wheel?  Here is my take on the classic Titanic Survival Prediction.  

## Step 1:  House Cleaning
### Tabula RASA to clear anything in my workspace and clean memory
```
##################################################
### TABULA RASA
##################################################
rm(list = ls(pattern = ''))
gc()
```

### Load the libraries required
```
##################################################
### LOAD LIBRARIES
##################################################
# MODELING FRAMEWORK
library(c(tidyverse, tidymodels, stacks, themis))

# DATA VISUALIZATION
library(c(ggplot2, ggcorrplot, vip))

# MODELING PACKAGES
library(c(glmnet, ranger, earth, xgboost, kknn, kernlab, keras))

# LOAD DATA
library(titanic)
```

### Create some functions that will be used later
```
##################################################
### LOAD FUNCTIONS
##################################################
# CREATE NUMERICAL STATS OUTPUT
# Quickly calculate basic numeric stats
numericStats <- function(x) {
     nCount <- round(sum(!is.na(x)), 0)
     nMissing <- round(sum(is.na(x)), 0)
     meanX <- round(mean(na.omit(x)), 4)
     sdX <- round(sd(na.omit(x)), 4)
     minX <- round(min(na.omit(x)), 4)
     tfPctl <- round(summary(x)[['1st Qu.']], 4)
     medianX <- round(median(x, na.rm = T), 4)
     sfPctl <- round(summary(x)[['3rd Qu.']], 4)
     maxX <- round(max(na.omit(x)), 4)
     return(data.frame(list(Metric = c('Count', 'Count Missing', 'Mean', 
                                    'Standard Deviation', 'Minimum', '1st Quartile', 
                                    'Median', '3rd Quartile', 'Maximum'),
                            Value = round(c(nCount, nMissing, meanX, 
                                            sdX, minX, tfPctl, 
                                            medianX, sfPctl, maxX), 0))))
}

# CRAMER'S V CALCULATION
# A function to calculate Cramer's V
cramersVTest <- function (x, y){
     cramersV <- sqrt(chisq.test(x, y, correct = FALSE)$statistic / 
                    (length(x) * 
                    (min(length(unique(x)), length(unique(y))) - 1)))
     return(as.numeric(cramersV))
}

# CONFUSION STATS
# Quickly calculates the fit stats from a confusion matrix
confusionStats <- function(pred, actu) {
     confMatrix <- table(pred, actu)
     tp <- confMatrix[2, 2]
     tn <- confMatrix[1, 1]
     fp <- confMatrix[1, 2]
     fn <- confMatrix[2, 1]
     tpr <- tp / (tp + fn)
     tnr <- tn / (tn + fp)
     ppv <- tp / (tp + fp)
     npv <- tn / (tn + fn)
     fnr <- fn / (fn + tp)
     fpr <- fp / (fp + tn)
     fdr <- fn / (fn + tp)
     fomr <- fn / (fn + tn)
     acc <- (tp + tn) / (tp + fp + tn + fn)
     bacc <- (tpr + tnr) / 2
     f1score <- (2 * tp) / ((2 * tp) + fp + fn)
     mcc <- sqrt(ppv * tpr * tnr * npv) - sqrt(fdr * fnr * fpr * fomr)
     return(data.frame(list(AKA = c('Sensitivity', 'Specificity', 'Precision', '',
                                    'Miss Rate', 'Fall-Out', '', '',
                                    '', '', '', ''),
                            Abbreviation = c('(TPR)', '(TNR)', '(PPV)', '(NPV)',
                                             '(FNR)', '(FPR)', '(FDR)', '(FOR)',
                                             '', '', '', '(MCC)'),
                            Metric = c('True Positive Rate', 'True Negative Rate ',
                                       'Positive Predictive Value', 'Negative Predictive Value',
                                       'False Negative Rate', 'False Positive Rate',
                                       'False Discovery Rate', 'False Omission Rate',
                                       'Accuracy', 'Balanced Accuracy', 'F1 Score',
                                       'Matthews Corr Coeff'),
                            Value = round(c(tpr, tnr, ppv, npv, fnr, fpr, fdr, fomr,
                                      acc, bacc, f1score, mcc), 4))))
}
```

This data set comes from the Titanic library.  The training and testing sets are already split, but I'm going to treat the train set like it is the full data set
```
##################################################
### LOAD DATA
##################################################
inputData <- titanic_train

#VALIDATE DATA
head(inputData)
colnames(inputData)
```

## Step 2:  Clean and Explore the data
I prefer to go through each and every variable, especially for my first check.  I check the class, look at a few rows of data, and do any type conversions and replacements necessary (characters to factors, characters to numbers/ints, replace 1/0 to 'Yes'/'No', replace NAs with zero, etc.)
```
##################################################
### CLEAN AND EXPLORE
##################################################
# PassengerId - Passenger ID number
summary(inputData$PassengerId)
class(inputData$PassengerId)
head(inputData$PassengerId, 25)
inputData$PassengerId <- as.factor(inputData$PassengerId)

# Survived - Survival (0 = No; 1 = Yes)
summary(inputData$Survived)
class(inputData$Survived)
head(inputData$Survived, 25)
inputData$Survived <- as.character(inputData$Survived)
inputData$Survived[inputData$Survived == 0] <- 'No'
inputData$Survived[inputData$Survived == 1] <- 'Yes'
inputData$Survived <- as.factor(inputData$Survived)

# Pclass - Passenger class (1 = 1st; 2 = 2nd; 3 = 3rd)
colnames(inputData)[colnames(inputData) == 'Pclass'] <- 'PClass'
summary(inputData$PClass)
class(inputData$PClass)
head(inputData$PPClass, 25)
inputData$PClass <- as.character(inputData$PClass)
inputData$PClass[inputData$PClass == 1] <- '1st Class'
inputData$PClass[inputData$PClass == 2] <- '2nd Class'
inputData$PClass[inputData$PClass == 3] <- '3rd Class'
inputData$PClass <- as.factor(inputData$PClass)

# Name - Passenger name
summary(inputData$Name)
class(inputData$Name)
head(inputData$Name, 25)

# Sex - Passenger sex
summary(inputData$Sex)
class(inputData$Sex)
head(inputData$Sex, 25)
inputData$Sex[inputData$Sex == 'male'] <- 'Male'
inputData$Sex[inputData$Sex == 'female'] <- 'Female'
inputData$Sex <- as.factor(inputData$Sex)

# Age - Passenger age
summary(inputData$Age)
class(inputData$Age)
head(inputData$Age, 25)

# SibSp - ### of sibblings/spouses aboard
summary(inputData$SibSp)
class(inputData$SibSp)
head(inputData$SibSp, 25)

# Parch - ### of parents / children aboard
summary(inputData$Parch)
class(inputData$Parch)
head(inputData$Parch, 25)

# Ticket - Ticket number
summary(inputData$Ticket)
class(inputData$Ticket)
head(inputData$Ticket, 25)

# Fare - Passenger fare (in British pounds)
summary(inputData$Fare)
class(inputData$Fare)
head(inputData$Fare, 25)

# Cabin - Cabin
summary(inputData$Cabin)
class(inputData$Cabin)
head(inputData$Cabin, 25)
inputData$Cabin[inputData$Cabin == ''] <- NA
inputData$Cabin <- as.factor(inputData$Cabin)

     # Multiple Cabins
     inputData$numberOfCabins <- str_count(inputData$Cabin, ' ') + 1
     inputData$numberOfCabins[is.na(inputData$Cabin)] <- 0
     summary(inputData$numberOfCabins)
     class(inputData$numberOfCabins)
     head(inputData$numberOfCabins, 25)
     
     # Cabin Location
     inputData$CabinLocation <- substring(inputData$Cabin, 1, 1)
     inputData$CabinLocation[is.na(inputData$CabinLocation)] <- 'U'
     inputData$CabinLocation <- as.factor(inputData$CabinLocation)
     summary(inputData$CabinLocation)
     class(inputData$CabinLocation)
     head(inputData$CabinLocation, 25)

# Embarked - Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
summary(inputData$Embarked)
class(inputData$Embarked)
head(inputData$Embarked, 25)
inputData$Embarked[inputData$Embarked == 'C'] <- 'Cherbourg'
inputData$Embarked[inputData$Embarked == 'Q'] <- 'Queenstown'
inputData$Embarked[inputData$Embarked == 'S'] <- 'Southampton'
inputData$Embarked[inputData$Embarked == ''] <- NA
inputData$Embarked <- as.factor(inputData$Embarked)
```

## Step 3:  Check the summary statistics for the numerical data
```
##################################################
### NUMERIC SUMMARY
##################################################
numericStats(inputData$Age)
numericStats(inputData$SibSp)
numericStats(inputData$Parch)
numericStats(inputData$Fare)
numericStats(inputData$numberOfCabins)
```

## Step 4:  Summarize the categorical data, specifically by the targeted variable
```
##################################################
### CATEGORICAL SUMMARY
##################################################
table(inputData$Survived)
table(inputData$PClass, inputData$Survived)
table(inputData$Sex, inputData$Survived)
table(inputData$CabinLocation, inputData$Survived)
table(inputData$Embarked, inputData$Survived)

round(prop.table(table(inputData$Survived)), digits = 2)
round(prop.table(table(inputData$PClass, inputData$Survived)), digits = 2)
round(prop.table(table(inputData$Sex, inputData$Survived)), digits = 2)
round(prop.table(table(inputData$CabinLocation, inputData$Survived)), digits = 2)
round(prop.table(table(inputData$Embarked, inputData$Survived)), digits = 2)
```

## Step 5:  Check for correlation in the numerical data
```
##################################################
### CORRELATION ANALYSIS
##################################################
# IDENTIFY NUMERIC COLUMNS 
colnames(inputData)
numericData <- inputData[, c(which(colnames(inputData) == 'Age'),
                             which(colnames(inputData) == 'SibSp'),
                             which(colnames(inputData) == 'Parch'),
                             which(colnames(inputData) == 'Fare'),
                             which(colnames(inputData) == 'numberOfCabins'))]
head(numericData)

# BUILD CORRELATION MATRIX AND P-VALUE MATRIX 
corrMatrix <- round(cor(numericData, use = 'pairwise.complete.obs'), 2) 
corrPValue <- round(cor_pmat(numericData, use = 'pairwise.complete.obs'), 4)

# BUILD CORRELATION MATRIX (CORRELOGRAM) 
ggcorrplot(corrMatrix, 
           ggtheme = ggplot2::theme_classic, 
           show.diag = TRUE, 
           p.mat = corrPValue, 
           insig = 'blank',
           type = 'lower', 
           outline.color = 'white', 
           lab = TRUE)
```

## Step 6:  Check for association in the categorical data
```
##################################################
### ASSOCIATION ANALYSIS
##################################################
# IDENTIFY CATEGORICAL COLUMNS 
colnames(inputData)
categoricalData <- inputData[, c(which(colnames(inputData) == 'PClass'),
                                 which(colnames(inputData) == 'Sex'),
                                 which(colnames(inputData) == 'CabinLocation'),
                                 which(colnames(inputData) == 'Embarked'),
                                 which(colnames(inputData) == 'Survived'))]
head(categoricalData)
categoricalData <- na.omit(categoricalData)

# VALIDATE ALL CATEGORICALS HAVE > 2 LEVELS
for (i in 1:length(colnames(categoricalData))){
     if (length(levels(categoricalData[[i]])) < 2){
          print(colnames(categoricalData[i]))
     }
     else{print('')}
}

# CREATE ASSOCIATION AND P-VALUE MATRIX
assocMatrix <- matrix(ncol = length(categoricalData), 
                      nrow = length(categoricalData), 
                      dimnames = list(names(categoricalData), 
                                      names(categoricalData)))
assocPValue <- assocMatrix

# FILL ASSOCIATION MATRIX
for (r in seq(nrow(assocMatrix))){
     for(c in seq(ncol(assocMatrix))){
          assocMatrix[[r, c]] <- suppressWarnings({round(
               cramersVTest(categoricalData[[r]], categoricalData[[c]]), 2)
          })
     }
}

# FILL ASSOCIATION P-VALUE MATRIX
for (r in seq(nrow(assocPValue))){
     for(c in seq(ncol(assocPValue))){
          assocPValue[[r, c]] <- suppressWarnings({round(
               chisq.test(categoricalData[[r]], categoricalData[[c]])$p.value, 4)
          })
     }
}

# BUILD ASSOCIATION MATRIX 
ggcorrplot(assocMatrix, 
           ggtheme = ggplot2::theme_classic, 
           show.diag = TRUE, 
           p.mat = assocPValue, 
           insig = 'blank',
           type = 'lower', 
           outline.color = 'white', 
           lab = TRUE)
```

## Step 7:  Check for correlation between all variables
To do this, you need to convert the categorical data to dummy variables (i.e., each category becomes it's own column).
```
##################################################
### DUMMY VARIABLE CORRELATION
##################################################
# IDENTIFY CATEGORICAL COLUMNS 
colnames(inputData)
dummyData <- inputData[, c(which(colnames(inputData) == 'Age'),
                           which(colnames(inputData) == 'SibSp'),
                           which(colnames(inputData) == 'Parch'),
                           which(colnames(inputData) == 'Fare'),
                           which(colnames(inputData) == 'numberOfCabins'),
                           which(colnames(inputData) == 'PClass'),
                           which(colnames(inputData) == 'Sex'),
                           which(colnames(inputData) == 'CabinLocation'),
                           which(colnames(inputData) == 'Embarked'),
                           which(colnames(inputData) == 'Survived'))]
head(dummyData)

# ONE-HOT ENCODE ALL VARIABLES
# Each category is created as it's own column; 
# Similarly to dummy variables in a regression model
dummyData <- model.matrix(~ 0 + ., data = na.omit(dummyData)) 

nrow(dummyData)
nrow(na.omit(inputData))
head(dummyData)
dummyData <- as.data.frame(dummyData)
dummyData$ID_Field <- na.omit(inputData[1:712, 'Name'])
aggregate(dummyData[,-c(which(colnames(inputData) == 'ID_Field'),
                        which(colnames(inputData) ==  'Fare'))] ~ ID_Field, FUN = sum)
rowsum(dummyData[,-c(which(colnames(inputData) == 'ID_Field'),
                     which(colnames(inputData) ==  'Fare'))],
       group = dummyData$ID_Field)
colnames(dummyData)
summary(dummyData$Age)
summary(dummyData$SibSp)
summary(dummyData$Parch)
summary(dummyData$Fare)
summary(dummyData$numberOfCabins)
summary(dummyData$SexMale)
summary(dummyData$CabinLocationB)
summary(dummyData$CabinLocationC)
summary(dummyData$CabinLocationD)
summary(dummyData$CabinLocationE)
summary(dummyData$CabinLocationF)
summary(dummyData$CabinLocationG)
summary(dummyData$CabinLocationT)
summary(dummyData$CabinLocationU)
summary(dummyData$EmbarkedQueenstown)
summary(dummyData$EmbarkedSouthampton)
summary(dummyData$SurvivedYes)



dummyMatrix <- round(cor(dummyData, use = 'pairwise.complete.obs'), 2)
dummyPValue <- round(cor_pmat(dummyData, use = 'pairwise.complete.obs'), 4)

# COPY DUMMY VARIABLE CORRELATION
write.table(dummyMatrix,
            file = "clipboard-128",
            sep = "\t",
            row.names = F,
            col.names = T)

# BUILD DUMMY VARIABLE CORRELATION MATRIX
ggcorrplot(dummyMatrix, 
           ggtheme = ggplot2::theme_classic, 
           show.diag = TRUE, 
           p.mat = dummyPValue, 
           insig = 'blank',
           type = 'lower', 
           outline.color = 'white', 
           lab = TRUE)
```

## Step 8:  Prepare the data for modeling
Create a new dataframe with the selected variables, omit NA's (if necessary), filter down the dataset (if necessary), split the data into training/testing sets, build the pre-processing recipe.
```
##################################################
### PREPARE DATA FOR MODELING
##################################################
# SUBSET DATA
finalData <- inputData[, c(which(colnames(inputData) == 'Age'),
                           which(colnames(inputData) == 'SibSp'),
                           which(colnames(inputData) == 'Parch'),
                           which(colnames(inputData) == 'Fare'),
                           which(colnames(inputData) == 'numberOfCabins'),
                           which(colnames(inputData) == 'PClass'),
                           which(colnames(inputData) == 'Sex'),
                           which(colnames(inputData) == 'CabinLocation'),
                           which(colnames(inputData) == 'Embarked'),
                           which(colnames(inputData) == 'Survived'))]

# OMIT NAs 

# SPLIT DATA: TEST/TRAIN
splitData <- initial_split(data = finalData,
                           strata = Survived,
                           prop = 0.70)
trainData <- training(splitData)
testData <- testing(splitData)

# CROSS VALIDATION
trainCVSplits <- vfold_cv(data = trainData,
                          strata = Survived,
                          v = 5,
                          repeats = 1)

# PREPROCESS RECIPE
preprocRec <- recipe(Survived ~ .,
                     data = finalData) |>
                     step_normalize(all_numeric()) |>
                     step_impute_knn(all_predictors()) |>
                     step_dummy(all_nominal(), -all_outcomes()) |>
                     step_rose(Survived)
```

## Step 9:  Elastic Net
Note: early on I was abbreviating this at logit, so if I missed any references please let me know and I'll change them to elasticNet.  
```
##################################################
### BUILD MODELS - ELASTIC NET
##################################################
# MODEL SPECIFICATIONS 
elasticNetSpec <-
     logistic_reg(penalty = tune(),
                  mixture = tune()) |>
                  set_engine('glmnet') |>
                  set_mode('classification')

# HYPERPARAMETER GRID 
elasticNetGrid <-
     elasticNetSpec |>
          parameters() |>
          grid_regular(levels = 10)

# CREATE WORKFLOW  
elasticNetWFlow <-
     workflow() |>
          add_recipe(preprocRec) |>
          add_model(elasticNetSpec)

# BUILD MODELS 
elasticNetTuning <- tune_grid(elasticNetWFlow,
                              resamples = trainCVSplits,
                              grid = elasticNetGrid,
                              control = control_stack_grid(),
                              metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# OPTIMIZED WORKFLOW 
elasticNetOptimized <- select_best(elasticNetTuning, metric = 'pr_auc')
elasticNetWFlow <- finalize_workflow(elasticNetWFlow, elasticNetOptimized)

# FITTED MODEL
elasticNetFitted <- fit_resamples(elasticNetWFlow, 
                                  resamples = trainCVSplits, 
                                  control = control_stack_grid(), 
                                  metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# CONFUSION MATRIX
table(Prediction = map(elasticNetFitted$.predictions, '.pred_class')[[1]],
      Actual = map(elasticNetFitted$.predictions, 'Survived')[[1]])

# FIT STATISTICS
confusionStats(pred = map(elasticNetFitted$.predictions, '.pred_class')[[1]],
               actu = map(elasticNetFitted$.predictions, 'Survived')[[1]])

# FINALIZE MODEL
elasticNetFinal <- fit(elasticNetWFlow, finalData)

# VARIABLE IMPORTANCE PLOT
elasticNetFinal |>
     pull_workflow_fit() |>
     vip(geom = 'col',
         num_features = ncol(finalData) - 1)

# EXTRACT MODEL OBJECT
elasticNetObject <- pull_workflow_fit(elasticNetFinal)
```

## Step 10:  Random Forest
```
##################################################
### BUILD MODELS - RANDOM FOREST
##################################################
# MODEL SPECIFICATIONS 
rforestSpec <- rand_forest(mtry = tune(),
                           trees = tune()) |>
                           set_engine('ranger', importance = 'impurity') |>
                           set_mode('classification')

# HYPERPARAMETER GRID 
rforestGrid <-
     rforestSpec |>
          parameters() |>
          finalize(select(finalData, -Survived)) |>
          grid_max_entropy(size = 10)

# CREATE WORKFLOW  
rforestWFlow <-
     workflow() |>
          add_recipe(preprocRec) |>
          add_model(rforestSpec)

# BUILD MODELS 
rforestTuning <- tune_grid(rforestWFlow,
                           resamples = trainCVSplits,
                           grid = rforestGrid,
                           control = control_stack_grid(),
                           metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# OPTIMIZED WORKFLOW 
rforestOptimized <- select_best(rforestTuning, metric = 'pr_auc')
rforestWFlow <- finalize_workflow(rforestWFlow, rforestOptimized)

# FITTED MODEL
rforestFitted <- fit_resamples(rforestWFlow, 
                               resamples = trainCVSplits, 
                               control = control_stack_grid(), 
                               metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# CONFUSION MATRIX
table(Prediction = map(rforestFitted$.predictions, '.pred_class')[[1]],
      Actual = map(rforestFitted$.predictions, 'Survived')[[1]])

# FIT STATISTICS
confusionStats(pred = map(rforestFitted$.predictions, '.pred_class')[[1]],
               actu = map(rforestFitted$.predictions, 'Survived')[[1]])

# FINALIZE MODEL
rforestFinal <- fit(rforestWFlow, finalData)

# VARIABLE IMPORTANCE PLOT
rforestFinal |>
     pull_workflow_fit() |>
     vip(geom = 'col',
         num_features = ncol(finalData) - 1)

# EXTRACT MODEL OBJECT
rforestObject <- pull_workflow_fit(rforestFinal)
```

## Step 11:  Multivariate Adaptive Regression Splines (MARS) 
The term MARS is trademarked, so R's package is called Earth
```
##################################################
### BUILD MODELS - MARS MODEL
##################################################
# MODEL SPECIFICATIONS 
marsSpec <- mars(num_terms = tune(),
                 prod_degree = tune(),
                 prune_method = 'backward') |>
                 set_engine('earth') |>
                 set_mode('classification')

# HYPERPARAMETER GRID 
marsGrid <- expand.grid(num_terms = ceiling(seq(from = 2,
                                                to = ncol(trainData),
                                                by = ncol(trainData) / 10)),
#                                                to = nrow(trainData) ^ (1 / 2),
#                                                by = nrow(trainData) ^ (1 / 2) / 10)),
                        prod_degree = 1:3)

# CREATE WORKFLOW  
marsWFlow <-
     workflow() |>
          add_recipe(preprocRec) |>
          add_model(marsSpec)

# BUILD MODELS 
marsTuning <- tune_grid(marsWFlow,
                        resamples = trainCVSplits,
                        grid = marsGrid,
                        control = control_stack_grid(),
                        metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# OPTIMIZED WORKFLOW 
marsOptimized <- select_best(marsTuning, metric = 'pr_auc')
marsWFlow <- finalize_workflow(marsWFlow, marsOptimized)

# FITTED MODEL
marsFitted <- fit_resamples(marsWFlow, 
                            resamples = trainCVSplits, 
                            control = control_stack_grid(), 
                            metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# CONFUSION MATRIX
table(Prediction = map(marsFitted$.predictions, '.pred_class')[[1]],
      Actual = map(marsFitted$.predictions, 'Survived')[[1]])

# FIT STATISTICS
confusionStats(pred = map(marsFitted$.predictions, '.pred_class')[[1]],
               actu = map(marsFitted$.predictions, 'Survived')[[1]])

# FINALIZE MODEL
marsFinal <- fit(marsWFlow, finalData)

# VARIABLE IMPORTANCE PLOT
marsFinal |>
     pull_workflow_fit() |>
     vip(geom = 'col',
         num_features = ncol(finalData) - 1)

# EXTRACT MODEL OBJECT
marsObject <- pull_workflow_fit(marsFinal)
#    str(marsObject, max.level = 2)
#    marsObject$fit$coefficients
```

## Step 12:  Extreme Gradient Boosting (XG Boosting) 
```
##################################################
### BUILD MODELS - XG BOOSTING
##################################################
# MODEL SPECIFICATIONS 
xgboostSpec <- boost_tree(trees = 1000,
                          tree_depth = tune(),
                          min_n = tune(),
                          loss_reduction = tune(),
                          sample_size = tune(),
                          mtry = tune(),
                          learn_rate = tune()) |>
                          set_engine('xgboost') |>
                          set_mode('classification')

# HYPERPARAMETER GRID 
xgboostGrid <-
     xgboostSpec |>
          parameters() |>
          finalize(select(finalData, -Survived)) |>
          grid_latin_hypercube(size = 10)

# CREATE WORKFLOW  
xgboostWFlow <-
     workflow() |>
          add_recipe(preprocRec) |>
          add_model(xgboostSpec)

# BUILD MODELS 
xgboostTuning <- tune_grid(xgboostWFlow,
                           resamples = trainCVSplits,
                           grid = xgboostGrid,
                           control = control_stack_grid(),
                           metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# OPTIMIZED WORKFLOW 
xgboostOptimized <- select_best(xgboostTuning, metric = 'pr_auc')
xgboostWFlow <- finalize_workflow(xgboostWFlow, xgboostOptimized)

# FITTED MODEL
xgboostFitted <- fit_resamples(xgboostWFlow, 
                               resamples = trainCVSplits, 
                               control = control_stack_grid(), 
                               metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# CONFUSION MATRIX
table(Prediction = map(xgboostFitted$.predictions, '.pred_class')[[1]],
      Actual = map(xgboostFitted$.predictions, 'Survived')[[1]])

# FIT STATISTICS
confusionStats(pred = map(xgboostFitted$.predictions, '.pred_class')[[1]],
               actu = map(xgboostFitted$.predictions, 'Survived')[[1]])

# FINALIZE MODEL
xgboostFinal <- fit(xgboostWFlow, finalData)

# VARIABLE IMPORTANCE PLOT
xgboostFinal |>
     pull_workflow_fit() |>
     vip(geom = 'col',
         num_features = ncol(finalData) - 1)

# EXTRACT MODEL OBJECT
xgboostObject <- pull_workflow_fit(xgboostFinal)
```

## Step 13:  K-Nearest Neighbor 
```
##################################################
### BUILD MODELS - K-NEAREST NEIGHBOR
##################################################
# MODEL SPECIFICATIONS 
knnSpec <- nearest_neighbor(neighbors = tune(),
                            weight_func = tune(),
                            dist_power = tune()) |>
                            set_engine('kknn') |>
                            set_mode('classification')

# HYPERPARAMETER GRID 
knnGrid <-
     knnSpec |>
          parameters() |>
          finalize(select(finalData, -Survived)) |>
          grid_latin_hypercube(size = 5)

# CREATE WORKFLOW  
knnWFlow <-
     workflow() |>
          add_recipe(preprocRec) |>
          add_model(knnSpec)

# BUILD MODELS 
knnTuning <- tune_grid(knnWFlow,
                       resamples = trainCVSplits,
                       grid = knnGrid,
                       control = control_stack_grid(),
                       metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# OPTIMIZED WORKFLOW 
knnOptimized <- select_best(knnTuning, metric = 'pr_auc')
knnWFlow <- finalize_workflow(knnWFlow, knnOptimized)

# FITTED MODEL
knnFitted <- fit_resamples(knnWFlow, 
                           resamples = trainCVSplits, 
                           control = control_stack_grid(), 
                           metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# CONFUSION MATRIX
table(Prediction = map(knnFitted$.predictions, '.pred_class')[[1]],
      Actual = map(knnFitted$.predictions, 'Survived')[[1]])

# FIT STATISTICS
confusionStats(pred = map(knnFitted$.predictions, '.pred_class')[[1]],
               actu = map(knnFitted$.predictions, 'Survived')[[1]])

# FINALIZE MODEL
knnFinal <- fit(knnWFlow, finalData)

# VARIABLE IMPORTANCE PLOT
# Not available at this time

# EXTRACT MODEL OBJECT
knnObject <- pull_workflow_fit(knnFinal)
```

## Step 14:  Support Vector Machine (SVM)
SVM can be linear, polynomial, or radial.  I chose radial for this example. 
```
##################################################
### BUILD MODELS - SUPPORT VECTOR MACHINE
##################################################
# MODEL SPECIFICATIONS 
svmSpec <- svm_rbf(cost = tune(),
                   rbf_sigma = tune(),
                   margin = tune()) |>
                   set_engine('kernlab') |>
                   set_mode('classification')

# HYPERPARAMETER GRID 
svmGrid <-
     svmSpec |>
          parameters() |>
          grid_max_entropy(size = 10)

# CREATE WORKFLOW  
svmWFlow <-
     workflow() |>
          add_recipe(preprocRec) |>
          add_model(svmSpec)

# BUILD MODELS 
svmTuning <- tune_grid(svmWFlow,
                       resamples = trainCVSplits,
                       grid = svmGrid,
                       control = control_stack_grid(),
                       metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# OPTIMIZED WORKFLOW 
svmOptimized <- select_best(svmTuning, metric = 'pr_auc')
svmWFlow <- finalize_workflow(svmWFlow, svmOptimized)

# FITTED MODEL
svmFitted <- fit_resamples(svmWFlow, 
                           resamples = trainCVSplits, 
                           control = control_stack_grid(), 
                           metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# CONFUSION MATRIX
table(Prediction = map(svmFitted$.predictions, '.pred_class')[[1]],
      Actual = map(svmFitted$.predictions, 'Survived')[[1]])

# FIT STATISTICS
confusionStats(pred = map(svmFitted$.predictions, '.pred_class')[[1]],
               actu = map(svmFitted$.predictions, 'Survived')[[1]])

# FINALIZE MODEL
svmFinal <- fit(svmWFlow, finalData)

# VARIABLE IMPORTANCE PLOT
# Not available at this time

# EXTRACT MODEL OBJECT
svmObject <- pull_workflow_fit(svmFinal)
```

## Step 15:  Neural Net
This is a neural net, but requires Python, tensorflow, and keras.  It was a pain.  I would look for another way to test a neural net model for future iterations.
```
##################################################
### BUILD MODELS - NEURAL NET
##################################################
# SET CONDA ENVIRONMENT
use_condaenv('r-tensorflow')

# MODEL SPECIFICATIONS 
neuralnetSpec <- mlp(hidden_units = tune(),
                     penalty = tune(),
                     activation = 'softmax') |>
                     set_engine('keras', verbose = FALSE) |>
                     set_mode('classification')

# HYPERPARAMETER GRID 
neuralnetGrid <-
     neuralnetSpec |>
          parameters() |>
          grid_max_entropy(size = 10)

# CREATE WORKFLOW  
neuralnetWFlow <-
     workflow() |>
          add_recipe(preprocRec) |>
          add_model(neuralnetSpec)

# BUILD MODELS 
neuralnetTuning <- tune_grid(neuralnetWFlow,
                             resamples = trainCVSplits,
                             grid = neuralnetGrid,
                             control = control_stack_grid(),
                             metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# OPTIMIZED WORKFLOW 
neuralnetOptimized <- select_best(neuralnetTuning, metric = 'pr_auc')
neuralnetWFlow <- finalize_workflow(neuralnetWFlow, neuralnetOptimized)

# FITTED MODEL
neuralnetFitted <- fit_resamples(neuralnetWFlow, 
                                 resamples = trainCVSplits, 
                                 control = control_stack_grid(), 
                                 metrics = metric_set(bal_accuracy, roc_auc, kap, pr_auc))

# CONFUSION MATRIX
table(Prediction = map(neuralnetFitted$.predictions, '.pred_class')[[1]],
      Actual = map(neuralnetFitted$.predictions, 'Survived')[[1]])

# FIT STATISTICS
confusionStats(pred = map(neuralnetFitted$.predictions, '.pred_class')[[1]],
               actu = map(neuralnetFitted$.predictions, 'Survived')[[1]])

# FINALIZE MODEL
neuralnetFinal <- fit(neuralnetWFlow, finalData)

# VARIABLE IMPORTANCE PLOT
# Not available at this time

# EXTRACT MODEL OBJECT
neuralnetObject <- pull_workflow_fit(neuralnetFinal)
```

## Step 15:  Stack the models
AKA create an ensemble.  This is where I found CARET to be superior.
```
##################################################
### STACK MODELS
##################################################
# BUILD STACK 
dataStack <-
     stacks() |>
          add_candidates(elasticNetFitted) |>
          add_candidates(rforestFitted) |>
          add_candidates(marsFitted) |>
          add_candidates(xgboostFitted) |>
          add_candidates(svmFitted) 

# FIT STACK
modelStack <- blend_predictions(dataStack, metric = metric_set(pr_auc))
modelStack <- fit_members(modelStack)

# TEST STACK
autoplot(modelStack)
autoplot(modelStack, type = 'members')
autoplot(modelStack, type = 'weights')
collect_parameters(modelStack, 'elasticNetFitted')
collect_parameters(modelStack, 'rforestFitted')
collect_parameters(modelStack, 'marsFitted')
collect_parameters(modelStack, 'xgboostFitted')
collect_parameters(modelStack, 'svmFitted')
collect_parameters(modelStack, 'neuralnetFitted')

# CONFUSION MATRIX
predictStack <-
     testData |>
          bind_cols(predict(object = modelStack, new_data = testData, type = 'prob'))

predictStack$predMove <- 'No'
predictStack$predMove[predictStack$.pred_Yes >= 0.50] <- 'Yes'

stackConfMat <- table(Prediction = predictStack$predMove, 
                      Actual = predictStack$Survived)
stackConfMat

# FIT STATISTICS
confusionStats(pred = predictStack$predMove, actu = predictStack$Survived)

# ROC AUC
roc_auc(predictStack,
        truth = Survived,
        contains('.pred_No'))

roc_curve(predictStack, truth = Survived, contains('.pred_No')) |>
     ggplot(aes(x = 1 - specificity, y = sensitivity)) +
     geom_path() +
     geom_abline(lty = 3) +
     coord_equal() +
     theme_bw()
```

## Step 16:  Export the stack
```
##################################################
### EXPORT MODEL 
##################################################
# SAVE THE MODEL
saveRDS(modelStack, 'E:/R Projects/Tidymodels - Classification/TitanicModel.rds')
```

# STOP
# Start a new R terminal and use this next set of code for "production".

## Step 17:  Load libraries
```
##################################################
### LOAD LIBRARIES
##################################################
# MODELING FRAMEWORK
library(c(tidyverse, tidymodels, stacks, themis))

# DATA VISUALIZATION
library(c(ggplot2, ggcorrplot, vip))

# MODELING PACKAGES
library(c(glmnet, ranger, earth, xgboost, kknn, kernlab, keras))

# LOAD DATA
library(titanic)
```

## Step 18:  Import the model
```
##################################################
### IMPORT MODEL AND TEST 
##################################################
# LOAD MODEL
testModel <- readRDS('D:/R Projects/Tidymodels - Classification/TitanicModel.rds')
```

## Step 19:  Load data
```
##################################################
### LOAD DATA 
##################################################
# LOAD NEW TESTING DATA
inputData <- titanic_test 
```

## Step 20:  Clean data
```
##################################################
### CLEAN DATA 
##################################################
# PassengerId - Passenger ID number
summary(inputData$PassengerId)
class(inputData$PassengerId)
head(inputData$PassengerId, 25)
inputData$PassengerId <- as.factor(inputData$PassengerId)

# Survived - Survival (0 = No; 1 = Yes)
summary(inputData$Survived)
class(inputData$Survived)
head(inputData$Survived, 25)
inputData$Survived <- as.character(inputData$Survived)
inputData$Survived[inputData$Survived == 0] <- 'No'
inputData$Survived[inputData$Survived == 1] <- 'Yes'
inputData$Survived <- as.factor(inputData$Survived)

# Pclass - Passenger class (1 = 1st; 2 = 2nd; 3 = 3rd)
summary(inputData$Pclass)
class(inputData$Pclass)
head(inputData$Pclass, 25)
inputData$Pclass <- as.character(inputData$Pclass)
inputData$PClass[inputData$Pclass == 1] <- '1st Class'
inputData$PClass[inputData$Pclass == 2] <- '2nd Class'
inputData$PClass[inputData$Pclass == 3] <- '3rd Class'
inputData$PClass <- as.factor(inputData$PClass)
summary(inputData$PClass)
class(inputData$PClass)
head(inputData$PClass, 25)

# Name - Passenger name
summary(inputData$Name)
class(inputData$Name)
head(inputData$Name, 25)

# Sex - Passenger sex
summary(inputData$Sex)
class(inputData$Sex)
head(inputData$Sex, 25)
inputData$Sex[inputData$Sex == 'male'] <- 'Male'
inputData$Sex[inputData$Sex == 'female'] <- 'Female'
inputData$Sex <- as.factor(inputData$Sex)

# Age - Passenger age
summary(inputData$Age)
class(inputData$Age)
head(inputData$Age, 25)

# SibSp - ### of sibblings/spouses aboard
summary(inputData$SibSp)
class(inputData$SibSp)
head(inputData$SibSp, 25)

# Parch - ### of parents / children aboard
summary(inputData$Parch)
class(inputData$Parch)
head(inputData$Parch, 25)

# Ticket - Ticket number
summary(inputData$Ticket)
class(inputData$Ticket)
head(inputData$Ticket, 25)

# Fare - Passenger fare (in British pounds)
summary(inputData$Fare)
class(inputData$Fare)
head(inputData$Fare, 25)

# Cabin - Cabin
summary(inputData$Cabin)
class(inputData$Cabin)
head(inputData$Cabin, 25)
inputData$Cabin[inputData$Cabin == ''] <- NA
inputData$Cabin <- as.factor(inputData$Cabin)

     # Multiple Cabins
     inputData$numberOfCabins <- str_count(inputData$Cabin, ' ') + 1
     inputData$numberOfCabins[is.na(inputData$Cabin)] <- 0
     summary(inputData$numberOfCabins)
     class(inputData$numberOfCabins)
     head(inputData$numberOfCabins, 25)
     
     # Cabin Location
     inputData$CabinLocation <- substring(inputData$Cabin, 1, 1)
     inputData$CabinLocation[is.na(inputData$CabinLocation)] <- 'U'
     inputData$CabinLocation <- as.factor(inputData$CabinLocation)
     summary(inputData$CabinLocation)
     class(inputData$CabinLocation)
     head(inputData$CabinLocation, 25)

# Embarked - Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
summary(inputData$Embarked)
class(inputData$Embarked)
head(inputData$Embarked, 25)
inputData$Embarked[inputData$Embarked == 'C'] <- 'Cherbourg'
inputData$Embarked[inputData$Embarked == 'Q'] <- 'Queenstown'
inputData$Embarked[inputData$Embarked == 'S'] <- 'Southampton'
inputData$Embarked[inputData$Embarked == ''] <- NA
inputData$Embarked <- as.factor(inputData$Embarked)



```

## Step 21:  Use the model 
Create predictions on new data. 
```
##################################################
### PREDICT NEW DATA 
##################################################
# PREDICT NEW DATA
survProb <- predict(testModel, inputData, type = 'prob')
inputData$survProb <- pull(survProb, .pred_Yes)
summary(inputData$survProb)
class(inputData$survProb)

survClass <- predict(testModel, inputData, type = 'class')
inputData$survClass <- pull(survClass, .pred_class)
summary(inputData$survClass)
class(inputData$survClass)

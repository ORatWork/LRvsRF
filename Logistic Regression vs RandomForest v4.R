#############################################
##
## Logistic Regression vs Randon Forest
##
## ORatWork
##
## Date 20150301
##
## Used as part of the Something to discuss over a good glass of wine; 
## Accuracy vs Interpretability blogpost
##
##############################################

#empty R - workspace

rm(list=ls())

# load required Lib's

require(caret)
require(arm)
require(corrplot)
require(randomForest)
require(gridExtra)

# get the data
# data can be found on https://archive.ics.uci.edu/ml/

red <- read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', header = TRUE, sep = ';')
white <- read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', header = TRUE, sep = ';')

# prepare to merge the two data sets

red[, 'color'] <- 'red'
white[, 'color'] <- 'white'

df <- rbind(red, white)
df$color <- as.factor(df$color)

# classify the wines with a quality factor >= 6 as good, smaller than 6 as bad

good_ones <- df$quality >= 6
bad_ones <- df$quality < 6
df[good_ones, 'quality'] <- 'good'
df[bad_ones, 'quality'] <- 'bad'  
df$quality <- as.factor(df$quality)
dim(df)

# make a copy of the raw data to feed into the estimation process 

copies <- dummyVars(quality ~ ., data = df)
df_copy<- data.frame(predict(copies, newdata = df))
df_copy[, 'quality'] <- df$quality

# split data set in train and test set
# set the seed for reproducibility

set.seed(1234) 

trainIndices <- createDataPartition(df_copy$quality, p = 0.7, list = FALSE)
train <- df_copy[trainIndices, ]
test <- df_copy[-trainIndices, ]

numericColumns <- !colnames(train) %in% c('quality', 'color.red', 'color.white')

# feature selection via correlation matrix
# only numeric features 

correlationMatrix <- cor(train[, numericColumns])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.6)
colnames(correlationMatrix)[highlyCorrelated]

corrplot(correlationMatrix, order = "FPC")

#take out the columns that are highly correlated

wanted=!colnames(train) %in% colnames(correlationMatrix)[highlyCorrelated]
wanted<-wanted & numericColumns

# set fit control to 10 fold cross validation

fitControl <- trainControl(method = 'cv', number = 10)

# fit a logistic regression

fit_glm <- train(x = train[,wanted], y = train$quality,
                 method = 'glm',
                 preProcess = 'range',
		 family    = binomial, 
                 trControl = fitControl) 
predict_glm <- predict(fit_glm, newdata = test[,wanted])
confMat_glm <- confusionMatrix(predict_glm, test$quality, positive = 'good')
importance_glm <- varImp(fit_glm, scale = TRUE)

plot(importance_glm, main = 'Feature importance for Logistic Regression')

# fit a random forest

fit_rf <- train(x = train[, wanted], y = train$quality,
                 method = 'rf',
                 trControl = fitControl,
                tuneGrid = expand.grid(.mtry = c(2:6)),
                n.tree = 1000) 
predict_rf <- predict(fit_rf, newdata = test[, wanted])
confMat_rf	 <- confusionMatrix(predict_rf, test$quality, positive = 'good')
importance_rf <- varImp(fit_rf, scale = TRUE)

plot(importance_rf, main = 'Feature importance for Random Forest')

#summary of the models used

models <- resamples(list(GLM = fit_glm,
                         RF = fit_rf))
dotplot(models)

results <- summary(models)
grid.table(results$statistics$Accuracy)
grid.table(results$statistics$Kappa)

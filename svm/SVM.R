### Source code for SVM Implementation ###
### ITCS 6156 - Machine Learning       ###
### Lily Naoudom ID: 800648084         ###

#install and import Caret Package
install.packages("caret")
library(caret)

#import data
opt_df <- read.csv("optdigits_training.csv", sep =',', header = FALSE)

#check structure of data frame
str(opt_df)
head(opt_df)

#data slicing
set.seed(3033)
intrain <- createDataPartition(y= opt_df$V65, p=0.7, list= FALSE)
training <- opt_df[intrain,]
testing <- opt_df[-intrain,]

#check dimensions
dim(training); dim(testing);

#preprocessing & training
anyNA(opt_df)

#dataset summarized details
summary(opt_df)

#convert variables to factors
training[["V65"]] = factor(training[["V65"]])

#training the SVM model
trctrl <- trainControl(method = "repeatedcv", number=10, repeats=3)
set.seed(3233)

svm_Linear <- train(V65 ~., data = training, method = "svmLinear", 
                    trControl= trctrl,
                    preProcess= c("center", "scale"),
                    tuneLength= 10)

#trained SVM model result
svm_Linear

#test set prediction
test_pred <- predict(svm_Linear, newdata = testing)
test_pred

#check accuracy
confusionMatrix(test_pred, testing$V65)

#build and tune SVM classifier with different C values
grid <- expand.grid(C = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 5))
set.seed(3233)
svm_Linear_Grid <- train(V65 ~., data = training, method = "svmLinear", 
                         trControl = trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_Linear_Grid

test_pred_grid <- predict(svm_Linear_Grid, newdata = testing)
test_pred_grid

#check accuracy of grid
confusionMatrix(test_pred_grid, testing$V65)

#use SVM classifier with non-linear kernel (RBF)
set.seed(3233)
svm_Radial <- train(V65 ~., data = training, method = "svmRadial", 
                    trControl=trctrl,
                    preProcess=c("center", "scale"),
                    tuneLength=10)
svm_Radial

test_pred_Radial <- predict(svm_Radial, newdata = testing)
confusionMatrix(test_pred_Radial, testing$V65)
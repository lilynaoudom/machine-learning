## SUPPORT VECTOR MACHINES practice
## set wd

### Packages needed for this session 
pkgs <- c( 'ggplot2', 'kernlab', 'ROCR' )
install.packages( pkgs ) 

################
## A) Linear SVM
################

### A.1) Load data
load('linear1.RData')

# This file contains:
# - linear1.train       : matrix of size n1 x 3 (first column is x1, second column is x2, third column is y)
# - linear1.test.input  : matrix of size n2 x 2 (first column is x1, second column is x2, y is unknown)
# - linear1.data        : aggregated info (train+test) for visualization purposes (see A.2)

### A.2) Visualize the data
require( 'ggplot2' )
qplot( data=linear1.data, x.1, x.2, colour=factor(y), shape=factor(train) )

### A.3) Train a linear SVM
require( 'kernlab' )
linear1.svm <- ksvm( y ~ ., data=linear1.train, type='C-svc', kernel='vanilladot',C=0.001, scale=c() )


# General summary
linear1.svm
# Attributes that you can access
attributes(linear1.svm)

alpha(linear1.svm)
alphaindex(linear1.svm)
b(linear1.svm)

### A.4) Plot the model
plot( linear1.svm, data=linear1.train )

### A.5) Adding points of test on the graph
points( linear1.test.input[ sample.int(nrow(linear1.test.input),10), ], pch=4 )

### A.6) Prediction
linear1.prediction <- predict( linear1.svm, linear1.test.input )

### A.7) Look at accuracy
load('linear1Sol.RData')
# contains linear1.test.output - the true labels for the test set
  print(paste0('Accuracy: ', 100*sum( linear1.prediction == linear1.test.output )/length(linear1.test.output), '%'))

#################################
### A.8) Non-separable datasets and accuracy 
#################################


### A.9) load the data
load('linear2.RData')

# This file contains:
# - linear2.data
# - linear2.train
# - linear2.test.input

### A.10) Visualize the data
require( 'ggplot2' )
qplot( data=linear2.data, x.1, x.2, colour=factor(y), shape=factor(train) )

### A.11) Train a linear SVM
require( 'kernlab' )
linear2.svm <- ksvm( y ~ ., data=linear2.train, type='C-svc', kernel='vanilladot',C=100, scale=c() )

### A.12) Plot the model
plot( linear2.svm, data=linear2.train )

### A.13) Prediction
linear2.prediction <- predict( linear2.svm, linear2.test.input)

### A.14) Look at accuracy
load('linear2Sol.RData')
# contains linear2.test.output

print(paste0('Accuracy: ', 100*sum( linear2.prediction == linear2.test.output )/length(linear2.test.output), '%'))

### A.15) A confusion matrix gives more information than just accuracy
print('Confusion Matrix: ');print(table( linear2.prediction, linear2.test.output, dnn= c("prediction","reality") ))

### A.16) Even better: ROC and Precision-Recall curves

linear2.prediction.score <- predict( linear2.svm, linear2.test.input, type='decision' )

require( 'ROCR' )

## ROC
linear2.roc.curve <- performance( prediction( linear2.prediction.score, linear2.test.output ), measure='tpr', x.measure='fpr' )
plot( linear2.roc.curve )

## Supplementary PR Curves
linear2.pr.curve <- performance( prediction( linear2.prediction.score, linear2.test.output ), measure='prec', x.measure='rec' )
plot( linear2.pr.curve )

##################################
###  B) Nonlinear SVM
##################################

### B.1) Load the data
load('nonlinear.RData')

### B.2) Visualize the data
qplot( data=nonlinear.data, x.1, x.2, colour=factor(y), shape=factor(train) )

### B.3) Let's still try a linear svm
badlinear.svm <- ksvm( y ~ ., data=nonlinear.train, type='C-svc', kernel='vanilladot', C=100, scale=c() )
plot( badlinear.svm, data=nonlinear.train )

### B.4) Ok try another one (RBF)
nonlinear.svm <- ksvm( y ~ ., data=nonlinear.train, type='C-svc', kernel='rbf', kpar=list(sigma=1), C=100, scale=c() )
plot( nonlinear.svm, data=nonlinear.train )


install.packages('manipulate')
require('manipulate')
manipulate( plot(ksvm( y ~ ., data=nonlinear.train, type='C-svc', kernel=k, C=2^c.exponent, scale=c() ), data=nonlinear.train ), c.exponent=slider(-10,10),
            k=picker('Gaussian'='rbfdot', 'Linear'='vanilladot', 'Hyperbolic'='tanhdot','Spline'='splinedot', 'Laplacian'='laplacedot') )

# Based on the manipulation above choose a kernel and a C value to set for the model
# and evaluate ther performnce 
nonlinear.svm <- ksvm( y ~ ., data=nonlinear.train, type='C-svc', kernel='rbf', kpar=list(sigma=1), C=100, scale=c() )
plot( nonlinear.svm, data=nonlinear.train )

### Prediction
nonLinear.prediction <- predict( nonlinear.svm, nonlinear.test.input)

# Look at accuracy
load('nonlinearSol.RData')
# contains non.test.output

print(paste0('Accuracy: ', 100*sum( nonLinear.prediction == nonlinear.test.output )/length(nonlinear.test.output), '%'))

print('Confusion Matrix: ');print(table( nonLinear.prediction, nonlinear.test.output, dnn= c("prediction","reality") ))

### ROC and Precision-Recall curves

nonLinear.prediction.score <- predict( nonlinear.svm, nonlinear.test.input, type='decision' )

require( 'ROCR' )

## ROC
nonLinear.roc.curve <- performance( prediction( nonLinear.prediction.score, nonlinear.test.output ), measure='tpr', x.measure='fpr' )
plot( nonLinear.roc.curve )

## Supplementary PR Curves
nonLinear.pr.curve <- performance( prediction( nonLinear.prediction.score, nonlinear.test.output ), measure='prec', x.measure='rec' )
plot( nonLinear.pr.curve )

### B.6) Bias-Variance Tradeoff
BiasVarianceTradeoff <- function( dataset, cross=10, c.seq=2^seq(-10, 10), ... ) {
  err <- sapply( c.seq, function( c ) 
  {
    cross( ksvm( y ~ ., data=dataset, C=c, cross=cross, ...) )
  })
  return(data.frame( c=c.seq, error=err ))
}

qplot( c, error, data=BiasVarianceTradeoff( nonlinear.train, type='C-svc', kernel='rbfdot' ), geom='line', log='x' )


# Function to generate a (n, p) matrix of draws from a random variable
RandomMatrix <- function( dist, n, p, ... ) {
    rs <- dist( n*p, ... )
    matrix( rs, n, p )
}

###################################################
#### Linear dataset
###################################################

# Function to generate a linearly separable dataset
GenerateDatasetLinear <- function( n, p, mean1=0, mean2=4 ) {
    n.pos <- floor( n/2 )
    x.pos <- RandomMatrix( rnorm, n.pos, p, mean=mean1, sd=1 )
    x.neg <- RandomMatrix( rnorm, n-n.pos, p, mean=mean2, sd=1 )
    y <- c( rep( 1, n.pos ), rep( -1, n-n.pos ) )
    n.train <- floor( 0.8 * n )
    idx.train <- sample( n, n.train )
    is.train <- rep( 0, n )
    is.train[idx.train] <- 1
    data.frame( x=rbind( x.pos, x.neg ), y=y, train=is.train )
}

# Generate the separable dataset
linear1.data <- GenerateDatasetLinear( 150, 2)

# Extract train and test subsets of the dataset
linear1.train <- linear1.data[linear1.data$train==1, ]
linear1.train <- subset( linear1.train, select=-train )

linear1.test <- linear1.data[linear1.data$train==0, ]

linear1.test.input <- subset( linear1.test, select=-y )
linear1.test.output <- linear1.test$y

linear1.data[linear1.data$train==0, "y"] <- NA

save(list=c('linear1.data', 'linear1.train', 'linear1.test.input'), file="linear1.RData")
save(linear1.test.output, file="linear1Sol.RData")

# Generate the separable dataset
linear2.data <- GenerateDatasetLinear( 500, 2, mean2=1)

# Extract train and test subsets of the dataset
linear2.train <- linear2.data[linear2.data$train==1, ]
linear2.train <- subset( linear2.train, select=-train )

linear2.test <- linear2.data[linear2.data$train==0, ]
linear2.test.input <- subset( linear2.test, select=-y )
linear2.test.output <- linear2.test$y

linear2.data[linear2.data$train==0, "y"] <- NA

# Saving
save(list=c('linear2.data', 'linear2.train', 'linear2.test.input'), file="linear2.RData")
save(linear2.test.output, file="linear2Sol.RData")


###################################################
#### Cross-shaped dataset
###################################################

# Generate a 'cross-shaped' dataset (not linearly separable)
GenerateDatasetNonlinear <- function( n, p ) {
    bottom.left <- RandomMatrix( rnorm, n, p, mean=0, sd=1 )
    upper.right <- RandomMatrix( rnorm, n, p, mean=4, sd=1 )
    tmp1 <- RandomMatrix( rnorm, n, p, mean=0, sd=1 )
    tmp2 <- RandomMatrix( rnorm, n, p, mean=4, sd=1 )
    upper.left <- cbind( tmp1[,1], tmp2[,2] )
    bottom.right <- cbind( tmp2[,1], tmp1[,2] )
    y <- c( rep( 1, 2 * n ), rep( -1, 2 * n ) )
    idx.train <- sample( 4 * n, floor( 3.5 * n ) )
    is.train <- rep( 0, 4 * n )
    is.train[idx.train] <- 1
    data.frame( x=rbind( bottom.left, upper.right, upper.left, bottom.right ), y=y, train=is.train )
}

# Extract train and test datasets
nonlinear.data <- GenerateDatasetNonlinear( 100, 2 )
nonlinear.train <- nonlinear.data[nonlinear.data$train==1, ]
nonlinear.train <- subset( nonlinear.train, select=-train )
nonlinear.test <- nonlinear.data[nonlinear.data$train==0, ]
nonlinear.test <- subset( nonlinear.test, select=-train )

# Extract train and test subsets of the dataset
nonlinear.train <- nonlinear.data[nonlinear.data$train==1, ]
nonlinear.train <- subset( nonlinear.train, select=-train )

nonlinear.test <- nonlinear.data[nonlinear.data$train==0, ]
nonlinear.test.input <- subset( nonlinear.test, select=-y )
nonlinear.test.output <- nonlinear.test$y

nonlinear.data[nonlinear.data$train==0, "y"] <- NA

save(list=c('nonlinear.data', 'nonlinear.train', 'nonlinear.test.input'), file="nonlinear.RData")
save(nonlinear.test.output, file="nonlinearSol.RData")





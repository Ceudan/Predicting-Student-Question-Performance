# Predicting-Student-Question-Performance
I led a team of 3 to write an ensemble machine learning model with bagging, to predict whether a student can correctly anwser a diagnositic question. Data includes performance of the same student on other questions, and performance of other students. Our ensemble combined 3 distinct machine learning architectures (K-Nearest Neighbors, Item-Response Theory, Autoencoder) each with bagging to maximize accuracy. 

## Background


## Data
Data contains the responses of 542 students to 1774 diagnostic questions (response is correct or incorrect). It can be visualized as a sparse matrix with each row representing a particular student's responses, and each column representing the responses to a particular question. Only 7% of the matrix is observed (68,000 points). 75%, 10%, and 15% of observed points are used for train, valid and test respectively (ratios are predetermined). 

![image of sparese matrix representation of data. Rows = num students, columns = num questions](images/sparse_matrix.PNG)

## Architecture
Steps:
1. Main dataset is subsampled into bags.
2. ML algorithms are run on each bag.
3. Output per algorithm is equally weighted across bags.
4. Final model output is an adjustably weighted mean of algorithms.

![image of sparese matrix representation of data. Rows = num students, columns = num questions](images/Architecture.png)

### Ensemble and Bagging

#### Theory
The motivation behind this section lies is explained by bias-variance. The expected squared error loss, can be broken down into the following terms.

![Image of Equation showing Expected Squared error decomposed into bias, varience of y, and variance of t](images/bias_variance.png)

Y represents the output of our model, while T represents the correct label. Y and T are random variables, with uncertaintly in Y mostly coming from our choice of train data, and uncertainty in T coming from inaccuracies in the labelling process. Below is a visualization for a 1 dimensional output.

![Image of 2 labelled bell curves, 1 for Y and another for T](images/bias_variance_plot.png)

Var(t), or the Bayes error, cannot be improved in our situation.

The bias is reduced via the ensemble. Inuitively we are averaging out the biases or incapabilities of each algorithm when trying to represent our desired function.

Var(y) is reduced via bagging. According to Bienaymé's formula, the mean of n random variables yi with the same covariance σ2 and correlation p is:

![Image of equation showing variance of the mean of N random variables as a function of N and their correlation](images/var_of_mean.png)

Inuitively this makes sense. If our bags are perfectly identical (p = 1), then our outputs are identical and we receive no variance reduction benefits. If our bags are completely independant (p = 0) we receive the full variance reduction benefits.

#### Hyperparameters

### K-Nearest Neighbors

### Item Response Theory

### Autoencoder

## Results

## Improvements 

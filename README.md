# Predicting-Student-Question-Performance
I led a team of 3 to write an ensemble machine learning model with bagging, to predict whether a student can correctly anwser a diagnositic question. Data includes performance of the same student on other questions, and performance of other students. Our ensemble combined 3 distinct machine learning architectures (K-Nearest Neighbors, Item-Response Theory, Autoencoder) each with bagging to maximize accuracy. 

## Background


## Data
Data contains the responses of 542 students to 1774 diagnostic questions (response is correct or incorrect). It can be visualized as a sparse matrix with each row representing a particular student's responses, and each column representing the responses to a particular question. Only 7% of the matrix is observed (68,000 points). 75%, 10%, and 15% of observed points are used for train, valid and test respectively (ratios are predetermined). 

![image of sparese matrix representation of data. Rows = num students, columns = num questions](images/sparse_matrix.PNG)

## Architecture

![image of sparese matrix representation of data. Rows = num students, columns = num questions](images/Architecture.png)

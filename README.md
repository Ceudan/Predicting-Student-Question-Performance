# Predicting-Student-Question-Performance
I led a team of 3 to write an ensemble machine learning model to predict whether a student can correctly anwser a specific diagnositic question. Data includes performance of the same student on other questions, and performance of other students. Our ensemble combined 3 distinct machine learning architectures (K-Nearest Neighbors, Item-Response Theory, Autoencoder) each with bagging to maximize accuracy. 
## Data
A subsample of the respones of 542 students to 1774 diagnostic questions is provided by Eedi (an online education platform). It can be visualled as a 2 dimensional sparse matrix.
![image of sparese matrix representation of data. Rows = num students, columns = num questions](images/sparse_matrix.png)

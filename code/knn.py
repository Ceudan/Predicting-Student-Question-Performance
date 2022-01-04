from sklearn.impute import KNNImputer
from utils import *
import numpy as np


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    # matrix is of the form (Item, User)
    mat = nbrs.fit_transform(matrix)
    # mat is of the form (Item, User_new), evalutate this.
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    # matrix is of the form (User, Item)
    mat = nbrs.fit_transform(matrix.T)
    # mat is of the form (User, Item_new), evalutate this.
    # Evalutation needs the form (Item_new, User) so transpose again.
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    return float(acc)


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    # Question 1a --------------------------------------------------------------|
    k_list = [1, 6, 11, 16, 21, 26]
    acc_list_user = [None]*len(k_list)
    for i in range(len(k_list)):
        k = k_list[i]
        acc_list_user[i] = knn_impute_by_user(matrix=sparse_matrix, valid_data=val_data, k=k)

    acc_arr_user = np.array(acc_list_user)
    k_star_user = k_list[np.argmax(acc_arr_user)]
    print(f"Best k: {k_star_user}")
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, k_star_user)
    print(f"Final Impute User Test Accuracy: {test_acc_user}")

    # Question 1c --------------------------------------------------------------|
    k_list = [1, 6, 11, 16, 21, 26]
    acc_list_item = [None]*len(k_list)
    for i in range(len(k_list)):
        k = k_list[i]
        acc_list_item[i] = knn_impute_by_item(matrix=sparse_matrix,
                                         valid_data=val_data,
                                         k=k)

    acc_arr_item = np.array(acc_list_item)
    k_star_item = k_list[np.argmax(acc_arr_item)]
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_star_item)
    print(f"Best k: {k_star_item}")
    print(f"Final Impute Item Test Accuracy:{test_acc_item}")

    # Question 1d --------------------------------------------------------------|
    # Compare user/item based collaborative filtering. Which performs better?
    # According to our results, conducting KNN by user gives a marginally better performance than KNN by item.
    # This is almost not noticable. The better performing user based KNN also uses far fewer neighbours improving scalability.

    # Question 1e --------------------------------------------------------------|
    # List two potential limitations of kNN.
    # 1. Dimensionality, computational cost scales with O(ND) for all queries.
    # 2. Normalization, the amount a feature on KNN changes by could greatly change
    # so the NN interpolation is not very accurate.


if __name__ == "__main__":
    main()

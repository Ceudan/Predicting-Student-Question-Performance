from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    log_lklihood = 0.

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c = data["is_correct"][i]
        x = (sigmoid((theta[u] - beta[q]).sum()))**c * \
            (1 - sigmoid((theta[u] - beta[q]).sum()))**(1-c)
        log_lklihood += np.log(x)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, iteration):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c = data["is_correct"][i]
        x = ((np.exp(beta[q]) / (np.exp(theta[u]) + np.exp(beta[q]))).sum())**c * \
            ((np.exp(theta[u]) / (np.exp(theta[u]) +
              np.exp(beta[q]))).sum())**(1-c)
        if iteration % 2 == 0:
            if c == 1:
                theta[u] += lr * x
            else:
                theta[u] += lr * -x
        else:
            if c == 1:
                beta[q] += lr * -x
            else:
                beta[q] += lr * x

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    neg_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        neg_lld_lst.append(neg_lld)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        if i < iterations - 1:
            theta, beta = update_theta_beta(data, lr, theta, beta, i)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.1
    iterations = 10
    theta, beta, val_acc_lst, neg_lld_lst = irt(
        train_data, val_data, lr, iterations)

    plt.subplot(2, 1, 1)
    plt.plot(range(1, iterations + 1), neg_lld_lst)
    plt.title("Train")
    plt.xlabel("Iteration")
    plt.ylabel("Neg. log-likelihood")

    plt.subplot(2, 1, 2)
    plt.plot(range(1, iterations + 1), val_acc_lst, color="orange")
    plt.title("Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Valdiation accuracy")

    plt.suptitle(
        "Training and validation neg. log-likelihoods")
    plt.tight_layout()
    plt.savefig("q2b.png", dpi=200)
    plt.show()

    print("\nHyperparameters:\nLearning rate = {}, Iterations = {}\n".format(
        lr, iterations))
    print("Final validation accuracy: {}\nFinal test accuracy: {}".format(
        val_acc_lst[-1], evaluate(test_data, theta, beta)))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    j_lst = np.random.randint(0, 1774, 3)

    for j in j_lst:
        pr_c_ij = []
        for i in range(len(theta)):
            pr_c_ij.append(sigmoid(theta[i] - beta[j]))
        plt.scatter(theta, pr_c_ij)

    plt.legend(j_lst, title="Question number")
    plt.title("Probability of the correct response given a question")
    plt.xlabel("Theta")
    plt.ylabel("P(c = 1)")
    plt.tight_layout()
    plt.savefig("q2d.png", dpi=200)
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

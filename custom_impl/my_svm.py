import cvxopt
import numpy as np
from cvxopt import solvers


class My_SVM():
    def __init__(self):
        pass
    def linear_kernel(self, x , y):
        return np.dot(x, y.T)
    def get_k_table(self, data):
        num_data = data.shape[0]

        K = np.zeros([num_data, num_data])

        for i in range(num_data):
            for j in range(num_data):
                K[i][j] = self.linear_kernel(data[i], data[j])

        return K
    def get_langrange_multipliers(self, data , labels):
        num_data, num_features = data.shape

        def get_P():
            Q = np.zeros([num_data, num_data])

            for i in range(num_data):
                for j in range(num_data):
                    Q[i][j] = labels[i] * labels[j] * self.linear_kernel(data[i], data[j])
                    # same with: (which is faster?)
                    # K_table = self.get_k_table(data)
                    # Q[i][j] = labels[i]*labels[j]*K_table[i][j]) ?

            return cvxopt.matrix(Q)  # cvxopt solvers require matrixes with doubles

        # we get the required arguments for cvxopt quadratic programming optimization
        P = get_P()
        q = cvxopt.matrix(np.ones(num_data) * -1)

        # we get the G and h matrixes for the inequality constraints
        # G must be a double matrix of size [num_data, num_data] for cvxopt
        G = cvxopt.matrix(np.diag(np.ones(num_data) * -1))
        h = cvxopt.matrix(np.zeros(num_data))

        solv = solvers.qp(P, q, G, h)

        print(solv)

        alphas = solv['x']
        return np.ravel(alphas)

    def get_b(self,support_vectors_labels,support_vectors):

        num_of_support_vectors = support_vectors_labels.shape[0]
        bs = 0
        for i in range(num_of_support_vectors):
            # for every support vector we find a b
            for j in range(num_of_support_vectors):
                sum = np.sum(support_vectors*support_vectors_labels*self.linear_kernel(support_vectors[i], support_vectors[j]))
                b = support_vectors_labels[i] - sum

            bs += b

        return (bs/num_of_support_vectors)
    def fit_data(self, data, labels):
        num_data, num_features = data.shape

        # we solve the quadratic programming problem to get the langrange multipliers
        alphas = self.get_langrange_multipliers(data, labels)

        # we find the support vectors from our dataset
        is_support_vector = alphas>0

        support_vectors = data[is_support_vector]
        support_vectors_labels = labels[is_support_vector]

        # we remove zero alphas we dont need
        alphas = alphas[is_support_vector]

        def decision_function(self, x_to_decide_for):
            sum =0.0

            for i in range(num_data):
                sum += support_vectors_labels[i]* alphas[i]*self.linear_kernel(support_vectors[i], x_to_decide_for)

            b = self.get_b(support_vectors_labels,support_vectors )

            return np.sign(sum + b)

        self.decision_function = decision_function

    def predict(self, x_to_decide_for):
        return self.decision_function(self,x_to_decide_for)


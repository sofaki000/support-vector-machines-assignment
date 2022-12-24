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
                    Q[i][j] = labels[i]*labels[j]*self.k_table[i][j]

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

    def get_b(self,support_vectors_labels):

        num_of_support_vectors = support_vectors_labels.shape[0]
        bs = 0
        for i in range(num_of_support_vectors):
            sum_of_sv = 0.0
            # for every support vector we find a b
            #for j in range(num_of_support_vectors):
            sum_of_sv += np.sum(self.alphas *
                                support_vectors_labels * self.k_table[self.support_vector_indexes[i],
                                self.support_vector_indexes])
            b = support_vectors_labels[i] - sum_of_sv

            bs += b

        return (bs/num_of_support_vectors)
    def fit_data(self, data, labels):
        k_table = self.get_k_table(data)
        self.k_table = k_table

        # we solve the quadratic programming problem to get the langrange multipliers
        alphas = self.get_langrange_multipliers(data, labels)

        # we find the support vectors from our dataset
        is_support_vector = alphas > 1e-5

        support_vectors = data[is_support_vector]

        self.support_vector_indexes = [i for i, x in enumerate(is_support_vector) if x]

        support_vectors_labels = labels[is_support_vector]

        # we remove zero alphas we dont need
        alphas = alphas[is_support_vector]
        self.alphas = alphas


        def decision_function(self, x_to_decide_for):
            sum =0.0
            support_vectors_num = support_vectors_labels.shape[0]
            multiplication = (support_vectors_labels*alphas)
            for i in range(support_vectors_num):
                sum += multiplication[i]* self.linear_kernel(support_vectors[i], x_to_decide_for)


            b = self.get_b(support_vectors_labels)

            return np.sign(sum + b)

        sv_num = support_vectors_labels.shape[0]

        self.decision_function = decision_function

        return sv_num

    def predict(self, x_to_decide_for):
        return self.decision_function(self,x_to_decide_for)


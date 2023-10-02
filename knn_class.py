import numpy as numpy

class KNN:
    def __init__(self, k=3):
        # define hyperparameter k
        self.k = k
    
    def fit(self, X, y):
        # store complete training data
        self.X_train - X
        self.y_train = y
        
    def predict(self, X_test):
        # get predictions for every row in test data
        y_pred = [self._get_single_prediction(x_test_row) for x_test_row in X_test]
        
    def _get_single_predicition(self, x_test_row):
        # get distances of a test_row vs all training rows
        distances = [self._get_euclidean_distance(x_test_row, x_train_row)
                     for x_train_row in self.X_train]
        # get indices of k-nearest-neighbours -> k-smallest distances
        k_index = np.argsort(distances)[:self.k]
        # get corresponding y-labels of training data
        k_labels = [self.y_train[index] for index in k_index]
        # return most common label
        return np.argmax(np.bincount(k_labels))
    
    def _get_euclidean_distance(self, x1, x2):
        # calculate the euclidean distance for a row pair
        sum_squared_distance = np.sum((x1 - x2)**2)
        return np.sqrt(sum_squared_distance)
    
    
    
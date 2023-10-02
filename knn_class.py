import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold

class KNN:
    def __init__(self, k):
        # define hyperparameter k
        self.k = k
    
    def fit(self, X, y):
        # store complete training data
        self.X_train = X
        self.y_train = y
        
    def predict(self, X_test):
        # get predictions for every row in test data
        y_pred = [self._get_single_prediction(x_test_row) for x_test_row in X_test]
        
    def _get_single_prediction(self, x_test_row):
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
    
    
if __name__ == '__main__':
    # define helper function to calculate accuracy
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred)/len(y_true)
        return accuracy
    
    # load dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # perform cross validation
    scores = []
    
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold, (index_train, index_valid) in enumerate(cv.split(X)):
        # split train and validation data
        X_train, y_train = X[index_train], y[index_train]
        X_valid, y_valid = X[index_valid], y[index_valid]
        
        k = 3
        clf = KNN(k=k)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_valid)
        
        score = accuracy(y_valid, predictions)
        scores.append(score)
print(scores)       
print(f'Mean Accuracy: {np.mean(scores)}')    
    
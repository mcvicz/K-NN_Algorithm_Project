import numpy as np

class KNNClassifier:
    """
    A simple implementation of the k-Nearest Neighbors classifier.
    
    This class supports only:
    - Euclidean distance
    - Classification (majority voting)
    
    It replicates the basic behavior of sklearn.neighbors.KNeighborsClassifier,
    but implemented manually for educational purposes.
    """

    def __init__(self, n_neighbors=3):
        """
        Parameters:
        n_neighbors : int
            Number of nearest neighbors used for voting.
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Stores training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training input samples.
        y : array-like, shape (n_samples,)
            Training labels.
        """
        # Convert to numpy arrays
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        """
        Computes Euclidean distance between two vectors.

        Parameters:
        x1, x2 : array-like
            Vectors to compare.

        Returns:
        float
            Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single(self, x):
        """
        Predicts label for one sample.

        Steps:
        1. Compute distance to every training point
        2. Sort distances
        3. Take k nearest neighbors
        4. Majority vote on labels

        Parameters:
        x : array-like
            Single input sample.

        Returns:
        label
            Predicted class label.
        """
        distances = []

        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        # sort by distance
        distances.sort(key=lambda tup: tup[0])

        # take k labels
        k_labels = [label for (_, label) in distances[:self.n_neighbors]]

        # majority vote
        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        """
        Predict class labels for multiple samples.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns:
        array-like
            Predicted labels.
        """
        X = np.array(X)
        predictions = []

        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)

        return np.array(predictions)

    def score(self, X, y):
        """
        Computes accuracy score.

        Parameters:
        X : array-like
            Test samples.
        y : array-like
            True labels.

        Returns:
        float
            Accuracy score between 0 and 1.
        """
        preds = self.predict(X)
        correct = np.sum(preds == np.array(y))
        return correct / len(y)

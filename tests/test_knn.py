import numpy as np
import pytest
import sys
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

# Add the parent directory to sys.path to locate the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knn_project.knn import KNNClassifier

# ---------------- FIXTURES (DATA) ----------------

@pytest.fixture
def simple_data():
    """Returns simple 2D data for quick testing."""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    y = np.array([0, 0, 1, 1])
    return X, y

@pytest.fixture
def iris_data():
    """Returns the real Iris dataset split into train/test sets."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# ---------------- UNIT TESTS ----------------

def test_initialization():
    """Checks if default and custom parameters work correctly."""
    model_default = KNNClassifier()
    assert model_default.n_neighbors == 3
    
    model_custom = KNNClassifier(n_neighbors=5)
    assert model_custom.n_neighbors == 5

def test_fit_storage(simple_data):
    """Checks if the fit method stores data correctly."""
    X, y = simple_data
    model = KNNClassifier()
    model.fit(X, y)
    
    assert np.array_equal(model.X_train, X)
    assert np.array_equal(model.y_train, y)

@pytest.mark.parametrize("point, expected_class", [
    (np.array([[1.1, 1.1]]), 0),   # Close to group 0
    (np.array([[10.1, 10.1]]), 1), # Close to group 1
    (np.array([[5.5, 5.5]]), 0),   # Middle (depends on K, assuming closer to 0 here)
])
def test_prediction_logic(simple_data, point, expected_class):
    """Checks prediction logic for various points (Parametrized)."""
    X, y = simple_data
    model = KNNClassifier(n_neighbors=1)
    model.fit(X, y)
    
    prediction = model.predict(point)
    assert prediction[0] == expected_class

def test_euclidean_math():
    """Checks mathematical correctness (Pythagorean theorem)."""
    model = KNNClassifier()
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    # Distance = sqrt(3^2 + 4^2) = sqrt(9+16) = 5
    assert model._euclidean_distance(p1, p2) == 5.0

# ---------------- INTEGRATION TESTS (Vs SKLEARN) ----------------

def test_compare_with_sklearn(iris_data):
    """
    Compares our implementation with Scikit-Learn on the Iris dataset.
    We expect at least 90% accuracy match.
    """
    X_train, X_test, y_train, y_test = iris_data
    K = 3

    # Our model
    my_model = KNNClassifier(n_neighbors=K)
    my_model.fit(X_train, y_train)
    my_preds = my_model.predict(X_test)

    # Sklearn model
    sk_model = SklearnKNN(n_neighbors=K, algorithm='brute') # brute force = matches our logic
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    # Check consistency/accuracy
    accuracy = np.mean(my_preds == sk_preds)
    print(f"\nMatch with Sklearn: {accuracy * 100:.2f}%")
    
    assert accuracy > 0.90, "Our model deviates too much from the Scikit-Learn baseline!"
import numpy as np
import pytest
import sys
import os

# To pozwala testom widzieć kod w folderze src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knn import KNNClassifier

# 1. Przygotowanie prostych danych (MOCK DATA)
@pytest.fixture
def dummy_data():
    # Dwie grupy punktów:
    # Grupa 0: (1,1), (1,2)
    # Grupa 1: (10,10), (10,11)
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    y = np.array([0, 0, 1, 1])
    return X, y

# 2. Test inicjalizacji
def test_init():
    knn = KNNClassifier(n_neighbors=5)
    assert knn.n_neighbors == 5

# 3. Test dopasowania (FIT)
def test_fit(dummy_data):
    X, y = dummy_data
    knn = KNNClassifier()
    knn.fit(X, y)
    
    assert np.array_equal(knn.X_train, X)
    assert np.array_equal(knn.y_train, y)

# 4. Test przewidywania (PREDICT)
def test_prediction(dummy_data):
    X, y = dummy_data
    knn = KNNClassifier(n_neighbors=1) # 1 sąsiad
    knn.fit(X, y)
    
    # Punkt (1.1, 1.1) jest blisko grupy 0
    assert knn.predict([[1.1, 1.1]])[0] == 0
    
    # Punkt (10.1, 10.1) jest blisko grupy 1
    assert knn.predict([[10.1, 10.1]])[0] == 1

# 5. Test matematyki (Dystans Euklidesowy)
def test_euclidean_distance():
    knn = KNNClassifier()
    # Trójkąt pitagorejski 3-4-5
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    
    dist = knn._euclidean_distance(p1, p2)
    assert dist == 5.0

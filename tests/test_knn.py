import numpy as np
import pytest
import sys
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

# Dodanie ścieżki do src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.knn import KNNClassifier

# ---------------- FIXTURES (DANE) ----------------

@pytest.fixture
def simple_data():
    """Zwraca proste dane 2D do szybkich testów."""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    y = np.array([0, 0, 1, 1])
    return X, y

@pytest.fixture
def iris_data():
    """Zwraca prawdziwy zbiór danych Iris (train/test)."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# ---------------- TESTY JEDNOSTKOWE ----------------

def test_initialization():
    """Sprawdza czy domyślne i niestandardowe parametry działają."""
    model_default = KNNClassifier()
    assert model_default.n_neighbors == 3
    
    model_custom = KNNClassifier(n_neighbors=5)
    assert model_custom.n_neighbors == 5

def test_fit_storage(simple_data):
    """Sprawdza czy metoda fit poprawnie zapisuje dane."""
    X, y = simple_data
    model = KNNClassifier()
    model.fit(X, y)
    
    assert np.array_equal(model.X_train, X)
    assert np.array_equal(model.y_train, y)

@pytest.mark.parametrize("point, expected_class", [
    (np.array([[1.1, 1.1]]), 0),   # Blisko grupy 0
    (np.array([[10.1, 10.1]]), 1), # Blisko grupy 1
    (np.array([[5.5, 5.5]]), 0),   # Środek (zależne od K, tu zakładamy bliżej 0)
])
def test_prediction_logic(simple_data, point, expected_class):
    """Sprawdza logikę predykcji dla różnych punktów (Parametrized)."""
    X, y = simple_data
    model = KNNClassifier(n_neighbors=1)
    model.fit(X, y)
    
    prediction = model.predict(point)
    assert prediction[0] == expected_class

def test_euclidean_math():
    """Sprawdza poprawność obliczeń matematycznych (Twierdzenie Pitagorasa)."""
    model = KNNClassifier()
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    # Dystans = sqrt(3^2 + 4^2) = sqrt(9+16) = 5
    assert model._euclidean_distance(p1, p2) == 5.0

# ---------------- TESTY INTEGRACYJNE (Vs SKLEARN) ----------------

def test_compare_with_sklearn(iris_data):
    """
    Porównuje naszą implementację z Scikit-Learn na zbiorze Iris.
    Oczekujemy co najmniej 90% zgodności wyników.
    """
    X_train, X_test, y_train, y_test = iris_data
    K = 3

    # Nasz model
    my_model = KNNClassifier(n_neighbors=K)
    my_model.fit(X_train, y_train)
    my_preds = my_model.predict(X_test)

    # Sklearn model
    sk_model = SklearnKNN(n_neighbors=K, algorithm='brute') # brute force = nasza logika
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    # Sprawdź zgodność
    accuracy = np.mean(my_preds == sk_preds)
    print(f"\nZgodność z Sklearn: {accuracy * 100:.2f}%")
    
    assert accuracy > 0.90, "Nasz model zbytnio odbiega od wzorca ze Scikit-Learn!"
import pandas as pd
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# KEY SETUP: Add 'src' folder to system path to import custom KNN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from knn_project.knn import KNNClassifier

# Set plot style
sns.set(style="whitegrid", font_scale=1.2)

def run_benchmarks():
    # Load datasets
    datasets = {
        "Iris": load_iris(),
        "Wine": load_wine(),
        "Breast Cancer": load_breast_cancer(),
        "Digits": load_digits()
    }

    K_values = [1, 3, 5, 7, 9]
    results = []

    print("Starting benchmark...")

    for name, data in datasets.items():
        print(f"Processing dataset: {name}...")
        X, y = data.data, data.target
        # Split data 70% train, 30% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        for K in K_values:
            # 1. Evaluate Custom Implementation
            start = time.perf_counter()
            knn = KNNClassifier(n_neighbors=K)
            knn.fit(X_train, y_train)
            train_time = time.perf_counter() - start

            start = time.perf_counter()
            knn.predict(X_test)
            predict_time = time.perf_counter() - start
            
            acc = knn.score(X_test, y_test)

            # 2. Evaluate Scikit-Learn Implementation
            start = time.perf_counter()
            sk_knn = KNeighborsClassifier(n_neighbors=K, algorithm='brute')
            sk_knn.fit(X_train, y_train)
            sk_train_time = time.perf_counter() - start

            start = time.perf_counter()
            sk_knn.predict(X_test)
            sk_predict_time = time.perf_counter() - start
            
            sk_acc = sk_knn.score(X_test, y_test)

            # Collect metrics
            results.append([
                name, K, acc, sk_acc, train_time, predict_time, sk_train_time, sk_predict_time
            ])

    # Create DataFrame
    df = pd.DataFrame(results, columns=[
        "Dataset", "K", "My Acc", "Sklearn Acc", 
        "My Train Time", "My Predict Time", "Sklearn Train Time", "Sklearn Predict Time"
    ])

    print("\n=== BENCHMARK RESULTS ===")
    print(df)
    
    # Save raw data
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to 'benchmark_results.csv'")

    # PLOTTING
    print("\nGenerating plots...")
    
    # Plot 1: Prediction Time Comparison (Log scale due to huge differences)
    plt.figure(figsize=(10, 6))
    df_melt = df.melt(id_vars=["Dataset", "K"], 
                      value_vars=["My Predict Time", "Sklearn Predict Time"],
                      var_name="Model", value_name="Time (s)")
    
    g = sns.catplot(data=df_melt, x="K", y="Time (s)", hue="Model", col="Dataset", kind="bar", height=4, sharey=False)
    g.set(yscale="log")
    plt.savefig("benchmark_prediction_times.png")
    print("Saved: benchmark_prediction_times.png")

    # Plot 2: Accuracy Comparison
    plt.figure(figsize=(10, 6))
    df_acc = df.melt(id_vars=["Dataset", "K"], 
                     value_vars=["My Acc", "Sklearn Acc"],
                     var_name="Model", value_name="Accuracy")
    
    g = sns.catplot(data=df_acc, x="K", y="Accuracy", hue="Model", col="Dataset", kind="bar", height=4)
    plt.ylim(0, 1.1)
    plt.savefig("benchmark_accuracy.png")
    print("Saved: benchmark_accuracy.png")

if __name__ == "__main__":
    run_benchmarks()
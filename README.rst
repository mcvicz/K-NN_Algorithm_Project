.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

======================
K-NN Algorithm Project
======================

Educational project aiming to implement the **K-Nearest Neighbors (K-NN)** algorithm from scratch and compare its performance with the `scikit-learn` library.

Authors and Credentials
=======================

* **Paweł Michalcewicz**: michalcewicz@student.agh.edu.pl
* **Krystian Ruszczak**: krystianr@student.agh.edu.pl

*Students of AGH University of Krakow*
*Course: Programming in Python Language*
*Tutor: Mustafa Sakhai*

Project Structure
=================

* ``src/knn_project/``
    Main package containing the ``KNNClassifier`` class implementation.
* ``tests/``
    Unit tests verifying the correctness of calculations (Euclidean distance, voting, fitting).
* ``notebooks/``
    Interactive Jupyter Notebook (``KNN_experiments.ipynb``) for experiments.
* ``run_benchmark.py``
    Main script automating the performance testing and report generation.
* ``benchmark_results.csv``
    CSV file containing raw results of time and accuracy measurements.
* ``docs/``
    Documentation source files (Sphinx).
* ``setup.cfg``
    Project configuration and dependencies (PyScaffold).

Installation and Configuration
==============================

To run the project, it is recommended to use a virtual environment.

1. Cloning the repository
-------------------------

.. code-block:: bash

    git clone https://github.com/mcvicz/K-NN_Algorithm_Project.git
    cd K-NN_Algorithm_Project

2. Installation (Editable Mode)
-------------------------------

This will install the project and all required dependencies (numpy, pandas, etc.) in your environment. Enable venv first.

.. code-block:: bash
    source venv/bin/activate
    pip install -e .

3. Running tests
----------------

Code quality is verified using ``pytest``.

.. code-block:: bash

    pytest

Expected result: **all tests green (PASSED)**.

Benchmark (Performance Comparison)
==================================

To compare our implementation with ``sklearn``, we use the Python script ``run_benchmark.py``.

.. code-block:: bash

    python run_benchmark.py

The script will:

1. Test both algorithms on 4 datasets (Iris, Wine, Breast Cancer, Digits).
2. Display the results table in the terminal.
3. Save results to ``benchmark_results.csv``.
4. Generate comparison plots (``benchmark_accuracy.png``, ``benchmark_prediction_times.png``).

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
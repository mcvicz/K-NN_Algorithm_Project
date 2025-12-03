import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

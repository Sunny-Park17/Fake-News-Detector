import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

#pulling data from csv files
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')
data_fake.head()
data_true.head()

data_fake["class"] = 0
data_true["class"] = 1

data_fake.shape, data_true.shape
print(data_fake.shape, data_true.shape)

dataf_manual_testing = data_fake.tail(10)

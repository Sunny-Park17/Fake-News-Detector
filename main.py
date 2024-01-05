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

dataF_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i], axis = 0, inplace = True)

dataT_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis = 0, inplace = True)

data_merge = pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Broder ="-"*40

########################################################################
#  Step 1 : Load the dataset
########################################################################

print(Broder)
print("Step 1 : Load the dataset")
print(Broder)

DatasetPath = "iris.csv"

df = pd.read_csv(DatasetPath)

print("Dataset gets loaded succesfully...")

print("Initial entries from dataset:")
print(df.head())

########################################################################
#  Step 2 : Data Analysis (EDA)
########################################################################

print(Broder)
print("Step 2 : Data Analysis")
print(Broder)

print("Shape of dataset : ",df.shape)
print("Column Names : ",list(df.columns))

print("Missing values (per Column)")
print(df.isnull().sum())    

print("Class Distribution (Species count)")
print(df["species"].value_counts())

print("Stastical Report of dataset")
print(df.describe())
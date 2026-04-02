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
print(df.head()) #tail

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

########################################################################
#  Step 3 : Decide Independent & Dependent variables
########################################################################

print(Broder)
print("Step 3 : Decide Independent & Dependent variables")
print(Broder)

# X : Independent variables / Features
# Y : Dependent variable / Feactures

feature_cols = [
    "sepal length (cm)",
    "sepal width  (cm)",
    "petal length  (cm)",
    "petal width  (cm)",
]

X = df[feature_cols]
Y = df["species"]

print("X shape : ",X.shape)
print("Y shape : ",Y.shape)
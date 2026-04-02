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

########################################################################
#  Step 4 : Visualisation of dataset
########################################################################

print(Broder)
print("Step 4 : Visualisation of dataset")
print(Broder)

# Scatter plot
plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal length  (cm)"],temp["petal width  (cm)"],label = sp)

plt.title("Iris : Petal lenght vs Petal width")
plt.xlabel("petal length  (cm)")
plt.ylabel("petal width  (cm)")
plt.legend()
plt.grid(True)
plt.show()

########################################################################
#  Step 5 : Split the dataset for training and testing
########################################################################

print(Broder)
print("Step 5 : Split the dataset for training and testing")
print(Broder)

# toral dataset = 150,5
# X = 150,4
# Y = 150,1
# Test Size = 20%
# Train Size = 80%

X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size=0.5,
    random_state=42,
)

print("Data spliting activity done : ")

print("X- Independent : ",X.shape)  # (150,4)
print("Y - Dependent  : ",Y.shape)  # (150,)

print("X_train : ",X_train.shape)   #(120,4)
print("X_test : ",X_test.shape)     #(30,4)

print("Y_train : ",Y_train.shape)  # (120,1)
print("Y_test : ",Y_test.shape)    #(30,1)

########################################################################
#  Step 6 : Bulid the model
########################################################################

print(Broder)
print("Step 6 : Bulid the model")
print(Broder)

print("We are going to use DecisionTreeClassifier")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42,
)

print("Model succesfully created : ",model)

########################################################################
#  Step 7 : Train the model
########################################################################

print(Broder)
print("Step 7 : Train the model")
print(Broder)

model.fit(X_train,Y_train)

print("Model training completed")

########################################################################
#  Step 8 : Evaluate the model
########################################################################

print(Broder)
print("Step 8 : Evaluate the model")
print(Broder)

Y_pred = model.predict(X_test)

print("Model evalution (testing) complete")

print(Y_pred.shape)

print("Expected answer : ")
print(Y_test)

print("Predicted answer :")
print(Y_pred)

########################################################################
#  Step 9 : Evaluate the model perfromnce
########################################################################

print(Broder)
print("Step 9 : Evaluate the model perfromnce")
print(Broder)

accuracy = accuracy_score(Y_test,Y_pred)

print("Accuracy of model is : ",accuracy*100)

cm = confusion_matrix(Y_test,Y_pred)

print("Confusion matix : ")
print(cm)

print("Classification Report")
print(classification_report(Y_test,Y_pred))
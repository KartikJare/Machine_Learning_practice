import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

def MarvellousClassifier(DataPath):
    broder = "-"*40

    # Step 1 : Load the dataset from CSV file

    print(broder)
    print("Step 1 : Load the dataset from CSV file")
    print(broder) 
    
    df = pd.read_csv(DataPath)
    print(broder)
    print("Some entries from the dataset : ")
    print(df.head)
    print(broder)

    # Step 2 : Clean the dataset by removing empty rows

    print(broder)
    print("Step 2 : Clean the dataset by removing empty rows")
    print(broder) 

    df.dropna(inplace=True)
    print("Total record : ",df.shape[0])
    print("Total Cloums : ",df.shape[1])
    
    # Step 3 : Separate indepemdent & Dependent Variable

    print(broder)
    print("Step 3 : Separate indepemdent & Dependent Variable")
    print(broder) 

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X : ",X.shape)
    print("Shape of Y : ",Y.shape)

    print(broder)
    print("Input cloumns : ",X.columns.tolist())
    print("Output cloums : Class")

    # Step 4 : Split the dataset for training and testing

    print(broder)
    print("Step 4 : Split the dataset for training and testing")
    print(broder)

    X_train,X_test,Y_tain,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

    print(broder)
    print("Information of training and testing data")
    print("X_train shape : ",X_train.shape)
    print("X_test shape : ",X_test.shape)
    print("Y_train shape : ",Y_tain.shape)
    print("Y_test shape : ",Y_test.shape)

    # Step 5 : Feature Scaling

    print(broder)
    print("Step 5 : Feature Scaling")
    print(broder)

    scalar = StandardScaler()
    # Independent variable scalling
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    print("Feature scaling is done")

    # Step 6 : Explore the multiple values of k 
    # Hyperparameter tuning (K)
    # Machine learn pipeline

    print(broder)
    print("Step 6 : Explore the multiple values of k")
    print(broder)

    accuracy_scores = []
    K_values = range(1,21)

    for k in K_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled,Y_tain)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test,Y_pred)
        accuracy_scores.append(accuracy)

    print(broder)
    print("Accuracy report of all K values from 1 to 20")
    for value in accuracy_scores:
        print(value)
    print(broder)

def main():

    broder = "-"*40
    print(broder)
    print("Wine Classifier using KNN")
    print(broder)

    MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()    
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

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

    

def main():

    broder = "-"*40
    print(broder)
    print("Wine Classifier using KNN")
    print(broder)

    MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()    
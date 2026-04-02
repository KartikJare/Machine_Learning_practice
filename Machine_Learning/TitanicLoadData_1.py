import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#--------------------------------------------------------------
#  Function     : DisplayInfo
#  Decscroption : It display the formated titled
#  Parameter    : title(str)
#  Return       : None
#  Date         : 14/03/26
#  Author       : Kartik Ganesh Jare
#--------------------------------------------------------------

def DisplayInfo(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

#--------------------------------------------------------------
#  Function     : showData
#  Decscroption : It show basic information about dataset
#  Parameter    : df
#                 df ->        pandas dataframe abject
#                 message
#                 message ->   Heading text to display
#  Return       : None
#  Date         : 14/03/26
#  Author       : Kartik Ganesh Jare
#----------------------------------------------------------------

def showData(df,message):
    DisplayInfo(message)

    print("\nFrist 5 rows of dataset")
    print(df.head())

    print("\nShape of dataset")
    print(df.shape)

    print("\nColumns name : ")
    print(df.columns.tolist())

    print("\nMissing values in each columns ")
    print(df.isnull().sum())

#--------------------------------------------------------------
#  Function     : MarvellousTitanicLogistic
#  Decscroption : This is main pipeline controller
#                 It loads the dataset,shows raw data 
#                 It perprocess the dataset & train the model
#  Parameter    : Data path of dataset file
#  Return       : None
#  Date         : 14/03/26
#  Author       : Kartik Ganesh Jare
#--------------------------------------------------------------

def MarvellousTitanicLogistic(DataPath):
    DisplayInfo("step 1 :Loading the dataset")
    df = pd.read_csv(DataPath)

    showData(df,"Initial datset")

#--------------------------------------------------------------
#  Function     : main 
#  Decscroption : Strating point of the function
#  Parameter    : None
#  Return       : None
#  Date         : 14/03/26
#  Author       : Kartik Ganesh Jare
#--------------------------------------------------------------

def main():
    MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")

if __name__ == "__main__":
    main()        
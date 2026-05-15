import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#--------------------------------------------------------------
#  Function     : loadPerserveModel
#  Decscroption : It is used to load preserved model
#  Parameter    : filenmae
#  Return       : model
#  Date         : 14/03/26
#  Author       : Kartik Ganesh Jare
#--------------------------------------------------------------
def loadPerserveModel(filename):

    loaded_model = joblib.load(filename)

    print("Model successfully loaded")

    return loaded_model

#--------------------------------------------------------------
#  Function     : PreserveModel
#  Decscroption : It is used to preserve model on secondary 
#  Parameter    : model,filenmae
#  Return       : None
#  Date         : 14/03/26
#  Author       : Kartik Ganesh Jare
#--------------------------------------------------------------

def PreserveModel(model,filename):
    joblib.dump(model,filename)
    
    print("Model preserved sucessfully with name : ",filename)

#--------------------------------------------------------------
#  Function     : TrainTitanicModel
#  Decscroption : It does split X,Y, training data,testing data
#  Parameter    : df
#  Return       : None
#  Date         : 14/03/26
#  Author       : Kartik Ganesh Jare
#--------------------------------------------------------------

def TrainTitanicModel(df):
    # Slipt features and labels
    X = df.drop("Survived",axis = 1)
    Y = df["Survived"]

    print("\nFeatures:")
    print(X.head())

    print("\nlable :")
    print(Y.head())

    print("shape of X : ",X.shape)
    print("shape of Y: ",Y.shape)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    print("X_train shape :",X_train.shape)
    print("X_test shape :",X_test.shape)
    print("Y_train shape :",Y_train.shape)
    print("Y_test shape :",Y_test.shape)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train,Y_train)

    print("Model training successfully")

    print("\nIntercept of model : ")
    print(model.intercept_)

    print("\nCoeeficent of model:")
    for feature,coeficent in zip(X.columns,model.coef_[0]):
        print(feature, " : ",coeficent)

    PreserveModel(model,"marvelloustitanic.pkl")    

    loaded_model = loadPerserveModel("marvelloustitanic.pkl")

    Y_pred = loaded_model.predict(X_test)

    accuracy = accuracy_score(Y_pred,Y_test)

    print("Accuracy is : ",accuracy)

    cm = confusion_matrix(Y_pred,Y_test)

    print("Confusion matrix :")
    print(cm)

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
    print(df.isnull().sum())\

#--------------------------------------------------------------
#  Function     : CleanTitanicData
#  Decscroption : It does perpricessing
#                 It removed unnecessary columns
#                 It handales missing values
#                 It covernts text data to numeric format
#                 It does encoding to categorical columns
#  Parameter    : df -> pandas dataframe
#  Return       : df -> Clean pandas dataframe 
#  Date         : 14/03/26
#  Author       : Kartik Ganesh Jare
#--------------------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original Data")
    print(df.head())

    # Remove unncessary columns
    drop_columns = ["Passengerid","zero","Name","Canbin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\n Columns to be dropped : ")
    print(existing_columns)

    # drop the unwanted columns
    df = df.drop(columns = existing_columns)
    DisplayInfo("step 2 : Data after colums removel")
    print(df.head())
    
    # Handle age columns
    if "Age" in df.columns:
        print("\nAge columns before filling missing values : ")
        print(df["Age"].head(10))

        # coerce -> Invaild value gets converted as NaN
        df["Age"] = pd.to_numeric(df["Age"],errors="coerce")  

        age_median = df["Age"].median() 

        # Replace missing values with median
        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge columns after preprocessing : ")
        print(df["Age"].head(10))

        # Handle Fare columns
        if "Fare" in df.columns:
            print("\nFare column before preprocessing : ")
            print(df["Fare"].head(10))
            
            df["Fare"] = pd.to_numeric(df["Fare"],errors="coerce") 
            
            fare_median = df["Fare"].median() 
            print("Meadin of fare columns is : ",fare_median)

            # Replace missing values with median
            df["Fare"] = df["Fare"].fillna(fare_median)

            print("\nFare columns after preprocessing : ")
            print(df["Fare"].head(10))

        # Handel Embarked columns
        if "Embarked" in df.columns:
            print("\nEmbarked column before preprocessing :")
            print(df["Embarked"].head(10))

            # covernt the data into string
            df["Embarked"] = df["Embarked"].astype(str).str.strip()

            # Remove missing values
            df["Embarked"] = df["Embarked"].replace(['nan','None',''],np.nan)

            # Get most frequent value
            embarked_mode = df["Embarked"].mode()[0]
            print("\nMode of embarked columns :",embarked_mode)

            df["Embarked"] = df["Embarked"].fillna(embarked_mode)

            print("\nEmbarked columns after preprocessing : ")
            print(df["Embarked"].head(10))

        # Handle Sex columns
        if "Sex" in df.columns:
            print("\nSex column before preprocessing : ")
            print(df["Sex"].head(10))
            
            df["Sex"] = pd.to_numeric(df["Sex"],errors="coerce") 

            print("\nSex columns after preprocessing : ")
            print(df["Sex"].head(10))

    DisplayInfo("Data After perprocessing")
    print(df.head())

    print("\nMissing Values after preprocessing:")
    print(df.isnull().sum())

    # Encode Embraked columns
    df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)
    print("\nData after encoding")

    print(df.head())

    print("Shape of dataset : ",df.shape)

    # convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\nData after encoding")

    print(df.head())

    print("Shape of dataset : ",df.shape)

    return df

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

    df = CleanTitanicData(df)

    TrainTitanicModel(df)

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
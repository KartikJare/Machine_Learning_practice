import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

def MarvellousAdvertise(DataPath):
    border = "-"*40
    #--------------------------------------------------
    #  Step 1 : Load dataset
    # -------------------------------------------------
    
    print(border)
    print("Step 1 : Load dataset")
    print(border)

    df = pd.read_csv(DataPath)

    print("Few records from the dataset : ")
    print(df.head())

    #--------------------------------------------------
    #  Step 2 : Remove unwanted columns
    # -------------------------------------------------

    print(border)
    print("Step 2 : Remove unwanted columns")
    print(border)

    print("Shape of the dataset before removel : ",df.shape)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'],inplace=True)

    print("Shape of the dataset after removel : ",df.shape)

    print(border)
    print("Clean dataset is : ")
    print(border)

    print(df.head())

    #--------------------------------------------------
    #  Step 3 : Check missing values
    # -------------------------------------------------

    print(border)
    print("Step 3 : Check missing values")
    print(border)
    
    print("Missing values count : \n",df.isnull().sum())

    #--------------------------------------------------
    #  Step 4 : Display Statistical summary
    # -------------------------------------------------

    print(border)
    print("Step 4 : Display Statistical summary")
    print(border)

    print(df.describe())

    #--------------------------------------------------
    #  Step 5 : Correlation between columns
    # -------------------------------------------------

    print(border)
    print("Step 5 : Correlation between columns")
    print(border)

    print("Correclation matrix :")
    print(df.corr())

    #--------------------------------------------------
    #  Step 6 : Split dataset into independent & dependent variables
    # -------------------------------------------------

    print(border)
    print("Step 6 : Split dataset into independent & dependent variables")
    print(border)

    X = df[['TV','radio','newspaper']]
    Y = df['sales']

    print("Shape of independent variables : ",X.shape)
    print("Shape of dependent variables : ",Y.shape)

    #--------------------------------------------------
    #  Step 7 : Split dataset for trainig & testing
    # -------------------------------------------------

    print(border)
    print("Step 7 : Split dataset for trainig & testing")
    print(border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    print("X_train shape : ",X_train.shape)
    print("X_test shape : ",X_test.shape)
    print("Y_train shape : ",Y_train.shape)
    print("Y_test shape : ",Y_test.shape)

    #--------------------------------------------------
    #  Step 8 : Create & train the model
    # -------------------------------------------------

    print(border)
    print("Step 8 : Create the model")
    print(border)

    model = LinearRegression()
    model.fit(X_train,Y_train)

    #--------------------------------------------------
    #  Step 9 : Test the model
    # -------------------------------------------------

    print(border)
    print("Step 9 :Test the model")
    print(border)

    Y_pred = model.predict(X_test)

    #--------------------------------------------------
    #  Step 10 : Evaluate the model
    # -------------------------------------------------

    print(border)
    print("Step 10 : Evaluate the model")
    print(border)    

    MSE = mean_squared_error(Y_test,Y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test,Y_pred)

    print("Mean square Error : ",MSE)
    print("Root Mean square Error : ",RMSE)
    print("R square value : ",R2)

    #--------------------------------------------------
    #  Step 11 : Calculate model coefficient
    # -------------------------------------------------

    print(border)
    print("Step 11 : Calculate model coefficient")
    print(border) 

    for colum,value in zip(X.columns,model.coef_):
        print(f"{colum} : {value}")

    print("Intercept : ",model.intercept_)

    #--------------------------------------------------
    #  Step 12 : Compare the actual and predicted values
    # -------------------------------------------------

    print(border)
    print("Step 12 : Compare the actual and predicted values")
    print(border) 

    Result = pd.DataFrame({
        'Actual sale' : Y_test.values,
        'Predicted sale': Y_pred
    })
    
    print(Result.head())

    #--------------------------------------------------
    #  Step 13 : Plot actual vs predicted
    # -------------------------------------------------

    print(border)
    print("Step 13 : Plot actual vs predicted")
    print(border)

    plt.figure(figsize=(8,5))
    plt.scatter(Y_test,Y_pred)
    plt.xlabel("Actual sales")
    plt.ylabel("predicted sales")
    plt.title("Actual sale vs perdicted sales")
    plt.grid(True)

    plt.show()
    
def main():
    MarvellousAdvertise("Advertising.csv")

if __name__ == "__main__":
    main()    
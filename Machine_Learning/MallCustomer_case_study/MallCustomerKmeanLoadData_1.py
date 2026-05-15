import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    #---------------------------------------------------------------
    #  Step 1 : Load the dataset
    #---------------------------------------------------------------

    print("Step 1 : Load the dataset")
    df = pd.read_csv("Mall_Customers.csv")

    print("Frist few records : ")
    print(df.head())

    print("Shape of dataset : ")
    print(df.shape)

    print("Missing value:")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()    
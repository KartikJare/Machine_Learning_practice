import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score

#------------------------------------------------------------------------------
# Step 1 : Load the dataset
#------------------------------------------------------------------------------

df = pd.read_csv("california_housing.csv")

print("Shape of dataset : ",df.shape)
print("Frist 5 records : ",df.head())

#------------------------------------------------------------------------------
# Step 2 : Separate fetures and lables
#------------------------------------------------------------------------------

X = df.drop("target",axis=1)
Y = df["target"]

#------------------------------------------------------------------------------
# Step 3 : Split dataset from training and testing 
#------------------------------------------------------------------------------

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#------------------------------------------------------------------------------
# Step 4 : Create Gradient Boosting model
#------------------------------------------------------------------------------

boost_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

#------------------------------------------------------------------------------
# Step 5 : Train Gradient Boosting model
#------------------------------------------------------------------------------

boost_model.fit(X_train,Y_train)

#------------------------------------------------------------------------------
# Step 6 : Test Gradient Boosting Bagging model
#------------------------------------------------------------------------------

Y_pred = boost_model.predict(X_test)

#------------------------------------------------------------------------------
# Step 7 : Evaluate Gradient Boosting model
#------------------------------------------------------------------------------

print("MeanSauaredError : ",mean_squared_error(Y_test,Y_pred))
print("R square : ",r2_score(Y_test,Y_pred))
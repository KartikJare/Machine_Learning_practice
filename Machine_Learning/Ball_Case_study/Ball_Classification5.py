# Steps for Machine Learning Application

# Step 1 : Data Gathering / Collection
# Step 2 : Data Analysis
# Step 3 : Data Cleaning
# Step 4 : Model selection 
# Step 5 : Model training   (Home Test)
# Step 6 : Model testing / Evaluation (Borad expale)
# Step 7 : Model Improvement    (Radio tune exmpale)
# Step 8 : Predicition / Deployment  (Last step)        

from sklearn import tree

# Rought = 1
# Smooth = 0

# Tennis = 1
# Cricket = 2

def main():
    print("Ball classification case study")

    # Original encoded dataset 
    # Independent Variables
    X = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]

    # Dependent Varaiables
    Y = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    # Independent varaiable for training
    Xtrain = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0]]
    
    # Independent varaiable for testing
    Xtest = [[35,1],[95,0]]

    # Dependent varaiable for training
    Ytrain = [1,1,2,1,2,1,2,1,1,1,2,1,2]

    # Dependent varaiable for testing
    Ytest = [1,2]
    
    moduleobj = tree.DecisionTreeClassifier()                 # module selection

    trainedmodule = moduleobj.fit(Xtrain,Ytrain)              # Trained

    Result = trainedmodule.predict(Xtest)                     #  Test 
    
    print("Model predicts the object as : ",Result)

if __name__ == "__main__":
    main()    

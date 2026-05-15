from sklearn.datasets import load_iris

def main():
    print("Iris classification case study")

    Dataset = load_iris()

    # Meta data of dataset
    print("Independent varaiables are : ")
    print(Dataset.feature_names)

    print("Depemdent variable are : ")
    print(Dataset.target_names)
    
if __name__ == "__main__":
    main()    

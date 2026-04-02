from sklearn.datasets import load_iris

def main():
    print("Iris classification case study")

    Dataset = load_iris()

    Broder = "-"*40

    print(Broder)

    for i in range(len(Dataset.target)):
        print("ID %d, Fetures %s , Lable %s" %(i,Dataset.data[i],Dataset.target[i]))
        
    print(Broder)
   
if __name__ == "__main__":
    main()    
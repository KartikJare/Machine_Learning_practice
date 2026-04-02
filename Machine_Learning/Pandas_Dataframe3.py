import pandas as pd

def main():
    Data = {
        "Name":["sagar","Amit","Pooja"],
        "Age":[23,26,25],
        "City":["Pune","Mumbai","Satara"]
    }

    dobj = pd.DataFrame(Data)

    print(dobj)
    
    print(dobj[["Name","City"]])

if __name__ == "__main__":
    main()    
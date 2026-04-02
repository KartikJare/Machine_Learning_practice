import pandas as pd

def main():
   
    sobj = pd.Series([25000,27000,29000,30000],index=["PPA","LB","Python","React"]) # new

    # bobj = pd.Series([1,2,"ppa","pune"])

    # print(bobj)

    print(sobj)

    print(sobj["Python"])

if __name__ == "__main__":
    main()    

# pandas.read_cvs
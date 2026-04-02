import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    
    dobj = pd.DataFrame(
        {
            "A":[1,2,3],
            "B":[4,5,6], #2 replcae
            "C":[7,8,9]
        }
    )

    print(dobj)

    # print(dobj.corr())

    # Feature Correlation
    sns.heatmap(dobj.corr(),annot=True)

    plt.show()


if __name__ == "__main__":
    main()    
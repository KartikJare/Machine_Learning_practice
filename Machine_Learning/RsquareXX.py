from sklearn.metrics import r2_score

def main():
    Y_Actual = [3,4,2,4,5]   # Y
    Y_predicated = [3,4,2,4,5]    # Yp

    r2 = r2_score(Y_Actual,Y_predicated)

    print("Actual values : Y ",Y_Actual)
    print("Perdicted values : Yp ",Y_predicated)
    print("R Square value : ",r2)   # 0.1

if __name__ == "__main__":
    main()    
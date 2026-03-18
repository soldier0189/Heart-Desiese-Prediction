import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def pre_processing(df: pd.DataFrame):

    df["Sex"] = df["Sex"].replace({'M':0,'F':1})
    df["ChestPainType"]= df["ChestPainType"].replace({"ASY":0,"NAP":1,"ATA":2,"TA":3})
    df["RestingECG"] = df["RestingECG"].replace({"Normal":0,"LVH":1,"ST":2})
    df["ExerciseAngina"] = df["ExerciseAngina"].replace({"N":0,"Y":1})
    df["ST_Slope"] = df["ST_Slope"].replace({"Flat":0,"Up":1,"Down":2})
    
    X = df.drop(columns=["HeartDisease"], axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    return [X_train, X_test, y_train, y_test] 

def save_data(df, path):
    dir_path = os.path.dirname(path)

    # create directory if not exists
    os.makedirs(dir_path, exist_ok=True)

    # save file
    df.to_csv(path, index=False)    

data_path = r"./data/raw/heart.csv"
x_train_file_path = r"./data/preprocessed/X_train.csv"
x_test_file_path = r"./data/preprocessed/X_test.csv"
y_train_file_path = r"./data/preprocessed/y_train.csv"
y_test_file_path = r"./data/preprocessed/y_test.csv"
def main():
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = pre_processing(df)
    save_data(X_train, x_train_file_path)
    save_data(X_test, x_test_file_path)
    save_data(y_train, y_train_file_path)
    save_data(y_test, y_test_file_path)
if __name__ == "__main__":
    main()
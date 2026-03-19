import pandas as pd
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
    
    return df

def save_data(df, path):
    dir_path = os.path.dirname(path)

    # create directory if not exists
    os.makedirs(dir_path, exist_ok=True)

    # save file
    df.to_csv(path, index=False)    

data_path = r"./data/raw/heart.csv"
save_file_path = r"./data/preprocessed/processed.csv"

def main():
    df = load_data(data_path)
    df = pre_processing(df)
    save_data(df, save_file_path)
    
if __name__ == "__main__":
    main()
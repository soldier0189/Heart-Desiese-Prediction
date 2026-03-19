import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def feature_engineering(df):
    X = df.drop(columns=["HeartDisease"], axis=1)
    y = df["HeartDisease"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    return [x_train, x_test, y_train, y_test]

def save_data(df, path):
    dir_path = os.path.dirname(path)

    # create directory if not exists
    os.makedirs(dir_path, exist_ok=True)

    # save file
    df.to_csv(path, index=False)

def main():
    file_path = r"./data/preprocessed/processed.csv"
    df = load_data(file_path)
    x_train, x_test, y_train, y_test = feature_engineering(df)
    
    data = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test
    }

    for name, dataset in data.items():
        save_file_path = f"./data/splited_data/{name}.csv"
        save_data(dataset, save_file_path)

if __name__ == "__main__":
    main()
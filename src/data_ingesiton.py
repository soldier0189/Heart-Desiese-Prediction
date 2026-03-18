import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def save_data(df, path):
    dir_path = os.path.dirname(path)

    # create directory if not exists
    os.makedirs(dir_path, exist_ok=True)

    # save file
    df.to_csv(path, index=False)

def main():
    file_path = r"./Notebook/heart.csv"
    save_path = r"./data/raw.heart.csv"
    df = load_data(file_path)
    save_data(df, save_path)

if __name__ == "__main__":
    main()
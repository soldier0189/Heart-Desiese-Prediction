import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)

def main():
   data = ["x_train", "x_test", "y_train", "y_test"]
   dataset = []
   for file in data:
         path = f"./data/splited_data/{file}.csv"
         dataset.append(load_data(path))

   x_train, x_test, y_train, y_test = dataset

   y_train = y_train.squeeze()
   y_test = y_test.squeeze()

   model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
   model.fit(x_train, y_train)
   save_path = "./artifacts/model.pkl"
   save_model(model, save_path)
if __name__ == "__main__":
    main()
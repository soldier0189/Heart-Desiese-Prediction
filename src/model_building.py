import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri("http://127.0.0.1:5000")


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
   
   mlflow.set_experiment("Heart desiese prediction")
   with mlflow.start_run() as run:
       run_id = run.info.run_id
       model = RandomForestClassifier(n_estimators=params["random_forest"]["n_estimators"], 
                                      max_depth=params["random_forest"]["max_depth"],
                                      random_state=params["random_forest"]["random_state"])
       model.fit(x_train, y_train)

       mlflow.log_params(params["random_forest"])
       mlflow.sklearn.log_model(model, "RandomForestClassifier")
       
       with open("run_id.txt", "w") as f:
        f.write(run_id)

       save_path = "./artifacts/model.pkl"
       save_model(model, save_path)
if __name__ == "__main__":
    main()
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import mlflow
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

with open("run_id.txt", "r") as f:
    run_id = f.read().strip()

def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
        return model

def load_data(path):
    df = pd.read_csv(path)
    return df

def main():
    model_path = "./artifacts/model.pkl"
    model = load_model(model_path)
    
    dataset = ["x_test", "y_test"]
    test_dataset = []
    for data in dataset:
        path = f"./data/splited_data/{data}.csv"
        test_dataset.append(load_data(path))
    
    x_test, y_test = test_dataset
    y_test = y_test.squeeze()

    y_pred = model.predict(x_test)
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)

        with open("confusion_matrix.json", "w") as f:
            json.dump(cm.tolist(), f)

        mlflow.log_artifact("confusion_matrix.json")
        
    
if __name__ == "__main__":
    main()


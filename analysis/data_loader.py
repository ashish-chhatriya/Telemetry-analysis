import kagglehub
import os
import pandas as pd

def load_dataset():
    path = kagglehub.dataset_download("colabsss/distributed-cloud-system-metrics")
    csv_path = os.path.join(path,"cloud_query_dataset.csv")
    df = pd.read_csv(csv_path)
    return df
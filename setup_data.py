import pandas as pd
import requests
import io
import os

def download_data():
    print("Downloading Telco Customer Churn dataset...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    response = requests.get(url)
    
    if response.status_code == 200:
        data_path = os.path.join(os.path.dirname(__file__), "data", "customer_churn.csv")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Read into pandas to ensure it's valid CSV and save
        df = pd.read_csv(io.StringIO(response.text))
        df.to_csv(data_path, index=False)
        print(f"Dataset successfully saved to {data_path}")
        print(f"Dataset shape: {df.shape}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    download_data()

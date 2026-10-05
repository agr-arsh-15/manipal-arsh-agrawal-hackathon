import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

api = KaggleApi()
api.authenticate()

raw_dir = "data/raw/benzinga"
os.makedirs(raw_dir, exist_ok=True)
ds = "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
print(f"Downloading {ds}...")
api.dataset_download_files(ds, path=raw_dir, unzip=True)
print("Benzinga files:", os.listdir(raw_dir))

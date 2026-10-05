import os
import zipfile
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

api = KaggleApi()
api.authenticate()

raw_dir = "data/raw"
os.makedirs(raw_dir, exist_ok=True)

# 1. Download FinancialPhraseBank
phrasebank_ds = "ankurzing/sentiment-analysis-for-financial-news"
phrasebank_dir = os.path.join(raw_dir, "phrasebank")
os.makedirs(phrasebank_dir, exist_ok=True)
print(f"Downloading {phrasebank_ds}...")
api.dataset_download_files(phrasebank_ds, path=phrasebank_dir, unzip=True)
print("PhraseBank files:", os.listdir(phrasebank_dir))

# 2. Download Stock Tweets dataset
tweets_ds = "thedevastator/tweet-sentiment-s-impact-on-stock-returns"
tweets_dir = os.path.join(raw_dir, "stock_tweets")
os.makedirs(tweets_dir, exist_ok=True)
print(f"Downloading {tweets_ds}...")
api.dataset_download_files(tweets_ds, path=tweets_dir, unzip=True)
print("Stock Tweets files:", os.listdir(tweets_dir))


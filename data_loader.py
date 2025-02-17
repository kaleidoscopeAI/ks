# data_loader.py
import numpy as np
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def crawl_webpage(url):
    """Crawls a single webpage and extracts the text content."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        page_text = ' '.join([p.text for p in soup.find_all('p')])
        logging.info(f"Successfully crawled webpage: {url}")
        return page_text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error crawling webpage {url}: {e}")
        return None

def load_data(csv_file='your_data.csv', use_web_crawl=False, crawl_url=None):
    """Loads data from CSV or web crawl."""
    try:
        if use_web_crawl:
            if not crawl_url:
                raise ValueError("crawl_url must be provided when use_web_crawl=True")
            webpage_content = crawl_webpage(crawl_url)
            if webpage_content:
                features = np.array([[webpage_content]])
                labels = np.array([0]) # Placeholder label
                data_chunk = {"features": features, "labels": labels}
                logging.info(f"Data chunk loaded from web crawl: {crawl_url} (CPU-based system).")
                return data_chunk
            else:
                logging.warning(f"Web crawling failed for URL: {crawl_url}. Returning None data chunk.")
                return None
        else: # Load from CSV
            df = pd.read_csv(csv_file)
            feature_columns = df.columns[:-1]
            label_column = df.columns[-1]
            X = df[feature_columns].values
            y = df[label_column].values
            data_chunk = {"features": X, "labels": y}
            logging.info(f"Data chunk loaded from CSV file: {csv_file} (CPU-based system).")
            return data_chunk
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found: {csv_file}")
        raise
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        data_from_csv = load_data()
        print("\nData Chunk from CSV:")
        if data_from_csv:
            print("Features shape:", data_from_csv['features'].shape)
            print("Labels shape:", data_from_csv['labels'].shape)
            print("First feature row:\n", data_from_csv['features'][0])
            print("First label:", data_from_csv['labels'][0])

        crawl_url_to_test = "https://www.example.com"
        data_from_web = load_data(use_web_crawl=True, crawl_url=crawl_url_to_test)
        print(f"\nData Chunk from Web Crawl of: {crawl_url_to_test}")
        if data_from_web:
            print("Features shape (web crawl data):", data_from_web['features'].shape)
            print("Labels shape (web crawl data):", data_from_web['labels'].shape)
            print("First feature (web crawl text):\n", data_from_web['features'][0][0][:200] + "...")
            print("First label (web crawl data):", data_from_web['labels'][0])
        else:
            print("Failed to load data from web crawl.")

    except FileNotFoundError:
        print("Error: your_data.csv not found.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Error loading data: {e}")

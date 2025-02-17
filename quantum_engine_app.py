import os
import joblib
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List

# Scrapy and BeautifulSoup imports for web scraping
import requests
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from scrapy.spider import Spider
from scrapy.selector import Selector
import scrapy
import urllib.parse  # For URL manipulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define Allowed Sites for Crawling ---
ALLOWED_SITES = [
    "pubchem.ncbi.nlm.nih.gov",  # PubChem - Chemical data
    "www.ebi.ac.uk/chembldb/",   # ChEMBL - Bioactivity data
    "www.drugbank.com",         # DrugBank - Drug data
    "www.genome.gov",           # National Human Genome Research Institute
    "clinicaltrials.gov",        # Clinical Trials data
    "www.fda.gov",              # FDA data
    "en.wikipedia.org",         # Wikipedia - for general info and scientific data
]

# --- Set Up Quantum Engine ---
class QuantumEngine:
    """Quantum-enhanced machine learning engine for advanced data analysis."""
    def __init__(self, model_type="RandomForest", model_path="quantum_model.pkl", test_size=0.2, scaler_type="StandardScaler"):
        self.model_type = model_type
        self.model_path = model_path
        self.test_size = test_size
        self.scaler_type = scaler_type
        self.model = None
        self.scaler = self._initialize_scaler()

    def _initialize_scaler(self):
        if self.scaler_type == "StandardScaler":
            return StandardScaler()
        elif self.scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")

    def _initialize_model(self):
        if self.model_type == "RandomForest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def initialize(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logging.info(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                logging.error(f"Error loading model from {self.model_path}: {e}")
                self.model = self._initialize_model()
        else:
            self.model = self._initialize_model()
            logging.info(f"No existing model found. Initialized a new {self.model_type} model.")

    def process_data(self, data_chunk: Dict, use_quantum_inspired=False) -> List[Dict]:
        try:
            X = data_chunk["features"]
            y = data_chunk["labels"]
            if self.model is None:
                raise ValueError("Model is not initialized. Call initialize() first.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            if not hasattr(self.model, "classes_"):
                self.model.fit(X_train, y_train)
                try:
                    joblib.dump(self.model, self.model_path)
                    logging.info(f"Trained and saved model to {self.model_path}")
                except Exception as e:
                    logging.error(f"Error saving model: {e}")

            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary")

            logging.info(f"Processed data chunk with accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, F1-score: {f1:.2f}")

            insights = []
            for i, pred in enumerate(predictions):
                if pred == 1:
                    insights.append({
                        "sample_id": i,
                        "prediction": "active",
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    })

            if use_quantum_inspired:
                logging.info("Applying quantum-inspired state propagation...")
                quantum_state = self._compute_quantum_state(X_test)
                for insight in insights:
                    insight["quantum_state"] = quantum_state

            return insights

        except Exception as e:
            logging.error(f"Error processing data chunk: {e}")
            raise

    def _compute_quantum_state(self, data: np.ndarray) -> np.ndarray:
        try:
            adjacency_matrix = np.corrcoef(data, rowvar=False)
            state_vector = np.random.random(data.shape[0])
            state_vector /= np.linalg.norm(state_vector)
            new_state = np.zeros_like(state_vector)
            for i in range(len(state_vector)):
                for j in range(len(state_vector)):
                    if adjacency_matrix[i, j] > 0:
                        new_state[i] += state_vector[j] * np.exp(-1j * adjacency_matrix[i, j])
            return new_state / np.linalg.norm(new_state)
        except Exception as e:
            logging.error(f"Error in quantum state propagation: {e}")
            raise

    def shutdown(self):
        logging.info("Quantum engine shut down.")

# --- Web Scraping Functions ---
def beautifulsoup_google_search_and_scrape(search_term, allowed_sites, num_results=3):
    try:
        search_query_with_sites = search_term + " " + " ".join([f"site:{site}" for site in allowed_sites])
        search_url = f"https://www.google.com/search?q={search_query_with_sites}"

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        search_results_list = []
        result_divs = soup.find_all('div', class_='Gx5Zad')

        for i, div in enumerate(result_divs[:num_results]):
            title_tag = div.find('h3')
            link_tag = div.find('a')
            snippet_tag = div.find('div', class_='BNeawe vvjwJb AP7Wnd')

            if title_tag and link_tag and snippet_tag:
                title = title_tag.text
                link = link_tag['href']
                snippet = snippet_tag.text
                search_results_list.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet,
                })

        return search_results_list if search_results_list else None

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error during BeautifulSoup web search: {e}")
        return None
    except Exception as e:
        logging.error(f"Error during BeautifulSoup web scraping: {e}")
        return None

# --- Scrapy Implementation ---
class DataSiteSpider(scrapy.Spider):
    name = 'data_site_spider'
    allowed_domains = ALLOWED_SITES
    start_urls = []

    def parse(self, response):
        item = {}
        item['title'] = response.css('title::text').get()
        item['link'] = response.url

        if "pubchem.ncbi.nlm.nih.gov" in response.url:
            description_selector = response.css('#Description-Description p::text')
            if description_selector:
                item['snippet'] = description_selector.get()
            else:
                item['snippet'] = "Description not found on PubChem page."

            properties_table = response.css('#Chemical-and-Physical-Properties')
            if properties_table:
                property_names = properties_table.css('dt::text').getall()
                property_values = properties_table.css('dd::text').getall()

                properties_dict = {}
                for name, value in zip(property_names, property_values):
                    properties_dict[name.strip()] = value.strip()

                item['data_content'] = properties_dict
            else:
                item['data_content'] = "Chemical Properties table not found on PubChem page."

        elif "en.wikipedia.org" in response.url: # Example for Wikipedia scraping - adapt as needed
            sections = response.css('div#mw-content-text div.mw-parser-output > h2, div#mw-content-text div.mw-parser-output > p') # Select sections and paragraphs in content
            content_text_list = []
            current_section_heading = "Introduction" # Default section
            section_content = ""

            for element in sections:
                if element.css('h2'): # If it's a section heading
                    if section_content: # Save previous section's content
                        content_text_list.append({current_section_heading: section_content.strip()})
                    current_section_heading = element.css('h2 span.mw-headline::text').get(default="Unnamed Section") # Get heading text
                    section_content = "" # Reset content for new section
                elif element.css('p'): # If it's a paragraph
                    section_content += element.css('p::text').get(default="") + "\n" # Append paragraph text

            if section_content: # Save the last section's content
                content_text_list.append({current_section_heading: section_content.strip()})

            item['data_content'] = content_text_list # Store list of sections and paragraphs
            item['snippet'] = "Extracted sections and paragraphs from Wikipedia page." # Basic snippet

        else:
            item['snippet'] = "Basic information scraped from allowed data site."
            item['data_content'] = "No specific data extraction rules defined for this site yet."

        yield item

def crawl_and_scrape(start_url):
    parsed_url = urllib.parse.urlparse(start_url)
    domain = parsed_url.netloc

    if domain not in ALLOWED_SITES:
        logging.warning(f"Crawling of '{start_url}' is not allowed. Domain '{domain}' is not in ALLOWED_SITES.")
        return None

    process = CrawlerProcess()
    spider_cls = DataSiteSpider
    spider_cls.start_urls = [start_url]
    crawler = process.create_crawler(spider_cls)
    results = []

    def item_scraped(item, response, spider):
        results.append(dict(item))

    crawler.signals.connect(item_scraped, signal=scrapy.signals.item_scraped)
    process.crawl(crawler)
    process.start()
    return results

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize the QuantumEngine
    engine = QuantumEngine(model_type="RandomForest", model_path="quantum_model.pkl")
    engine.initialize()

    search_term = "chemical properties of caffeine"
    search_results = beautifulsoup_google_search_and_scrape(search_term, ALLOWED_SITES)

    if search_results:
        print("Search Results from Allowed Sites (Google Search via BeautifulSoup4):")
        for result in search_results:
            print(result)

    print("\nCrawling PubChem for caffeine data (Scrapy Direct Crawl)...")
    crawl_results_pubchem = crawl_and_scrape("https://pubchem.ncbi.nlm.nih.gov/compound/caffeine")
    if crawl_results_pubchem:
        print("\nCrawl Results from PubChem (Scrapy Direct Crawl):")
        for result in crawl_results_pubchem:
            print(result)

    print("\nCrawling Wikipedia for caffeine data (Scrapy Direct Crawl)...")
    crawl_results_wiki = crawl_and_scrape("https://en.wikipedia.org/wiki/Caffeine")
    if crawl_results_wiki:
        print("\nCrawl Results from Wikipedia (Scrapy Direct Crawl):")
        for result in crawl_results_wiki:
            print(result)

    engine.shutdown()

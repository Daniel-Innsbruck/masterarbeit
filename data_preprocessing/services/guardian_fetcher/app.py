import requests
from pymongo import MongoClient
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import schedule
import logging

load_dotenv()

# --- KONFIGURATION ---
API_KEY = os.getenv('API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
LOG_PATH = "logs/job.log"

# create logging directory
log_dir = os.path.dirname(LOG_PATH)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

# logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def fetch_guardian_data(mode="daily"):
    logger.info(f"Starte Job (mode={mode})")
    client = MongoClient(MONGO_URI)
    db = client.guardian_db
    collection = db.articles

    if mode == "initial":
        from_date = "2026-01-02"
        to_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        yesterday = datetime.now() - timedelta(days=1)
        from_date = yesterday.strftime("%Y-%m-%d")
        to_date = yesterday.strftime("%Y-%m-%d")

    url = "https://content.guardianapis.com/search"
    page = 1
    total_pages = 1
    total_stored = 0

    print(f"Starte Download ab Datum: {from_date}...")

    while page <= total_pages:
        params = {
            'api-key': API_KEY,
            'from-date': from_date,
            'to-date': to_date,
            'show-fields': 'all',
            'show-tags': 'all',
            'page-size': 50,
            'page': page,
            'order-by': 'oldest'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()['response']

            if page == 1:
                total_pages = data['pages']
                logger.info(f"{data['total']} Artikel auf {total_pages} Seiten gefunden")

            articles = data['results']
            if not articles:
                logger.info("Keine weiteren Artikel gefunden")
                break

            from pymongo import UpdateOne
            ops = [
                UpdateOne({'id': art['id']}, {'$set': art}, upsert=True)
                for art in articles
            ]
            collection.bulk_write(ops)
            total_stored += len(articles)
            logger.info(f"Seite {page}/{total_pages} verarbeitet ({len(articles)} Artikel)")

            page += 1
            time.sleep(1)

        except Exception as e:
            logger.error(f"Fehler auf Seite {page}: {e}", exc_info=True)
            break

        logger.info(f"{total_stored} Artikel gespeichert/aktualisiert")
        logger.info("Job beendet\n")

def run_job():
    logger.info("===== JOB START =====")
    fetch_guardian_data(mode="daily")
    logger.info("===== JOB END =====")


if __name__ == "__main__":
    run_job()

    schedule.every().day.at("03:00").do(run_job)
    while True:
        schedule.run_pending()
        time.sleep(60)
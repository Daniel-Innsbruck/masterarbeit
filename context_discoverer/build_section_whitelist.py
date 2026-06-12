from pymongo import MongoClient
import json

client = MongoClient("mongodb://admin:sokba4-fyvtUs-nezcov@qe-ragevalk5.uibk.ac.at:27017/?authSource=admin")
db = client["guardian_db"]

ALLOWED_SECTIONS = ["world", "politics", "us-news", "uk-news", "australia-news"]

article_ids = []
for doc in db.articles.find({"sectionId": {"$in": ALLOWED_SECTIONS}}, {"id": 1, "_id": 0}):
    article_ids.append(doc["id"])

with open("../data/allowed_article_ids.json", "w") as f:
    json.dump(article_ids, f)

print(f"Exported {len(article_ids)} article IDs")
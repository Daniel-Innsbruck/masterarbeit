# Data Acquisition and Preprocessing Pipeline

This directory contains the scripts and documentation required to build the Evaluation Vector Database ($V_{eval}$) used in the RAG-DIVE multi-hop evaluation framework (see Thesis Section 4.2).

## Prerequisites
To replicate the data setup, the following tools are required:
* **Docker & Docker Compose:** To run the automated fetcher and the MongoDB instance.
* **Poetry:** To manage Python dependencies and the virtual environment.
* **API Keys:** A `.env` file must be created in the root directory containing `API_KEY` (The Guardian Open Platform) and the Google Gemini API key.

## Setup Instructions

### 1. Dependency Management
Initialse the virtual environment and install all required packages strictly using the provided `poetry.lock` file:
`poetry config virtualenvs.in-project true`
`poetry install`
`source .venv/bin/activate`

### 2. Data Acquisition (The Guardian Fetcher)
The `fetcher.py` script retrieves yesterday's Guardian articles from the Open Platform API as a daily cron job.
* To start the fetcher and the associated MongoDB instance, use the provided Docker setup:
`docker-compose up -d`

### 3. Vector Database Initialisation
Once the MongoDB is populated, execute the Jupyter Notebook `add_embeddings_to_guardian.ipynb`. 
* This notebook processes the raw articles via recursive character chunking.
* It embeds the chunks using the `gemini-embedding-001` model.
* It explicitly links each chunk to its parent document via the `article_id` metadata field, which is a strict requirement for the framework's stochastic multi-hop context discovery (Section 4.3).
* The output is a local ChromaDB instance stored in `./chroma_db`.
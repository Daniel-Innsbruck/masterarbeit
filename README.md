# DSIA_2024_Hillebrand_Daniel_MA

 **Automated Evaluation of RAG Systems for Multi-Turn, Multi-Hop Scenarios**

This repository accompanies the results and raw data from the Master's thesis **Automated Evaluation of RAG Systems for
Multi-Turn, Multi-Hop Scenarios"**. The work extends the original framework **“RAG-DIVE:  A Dynamic Approach for 
Multi-Turn Dialogue Evaluation in Retrieval-Augmented Generation”** of Brehme et al. (2026) by introducing advanced 
capabilities for multi-hop reasoning and cross-document context construction. The framework enables the automated 
evaluation of Retrieval-Augmented Generation (RAG) systems by simulating complex, document-bridging dialogues through a
two-stage context discovery mechanism, deterministic context expansion, and an integrated response-aware dialogue 
caching layer. It provides a robust testing environment for assessing factual integrity and reasoning performance in
dynamic, multi-turn interactions.

---

## Repository Overview

This repository contains the **data, code, and configurations** used for all experiments presented in the Master's thesis..

### Folder Structure

| Folder                         | Description                                                                                                                                                                                                                                                                                                                                                         |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`data/`**                    | Contains all generated conversations and corresponding logs. Each experiment folder includes:<br>• One JSON file per run (conversation data)<br>• Log file (Conversation Validator output)<br>• Metric files:<br> – *Multiturn metrics*: forgetfulness, context retention<br> – *Single-turn metrics*: correctness, faithfulness, context precision, context recall |
| **`data/industrial_usecase/`** | Includes the industrial use case experiments                                                                                                                                                                                                                                                                                                                        |
| **`human_validation/`**        | Contains Excel sheets with human validation results of generated conversations.<br>The main dataset used: `data/ADD_CORRECT_PATH_HERE`.<br>To view the conversation, open `viewer.html` in your browser.                                                                                                                                                            |
| **`context_discoverer/`**      | Implements the Context Discoverer (CD) for multi-hop semantic bridging via filtered KNN search and LLM validation.                                                                                                                                                                                                                                                  |
| **`conversation_generator/`**  | Code used for **synthetic conversation generation** with multiple model configurations and personas.                                                                                                                                                                                                                                                                |
| **`conversation_validator/`**  | Code for **automatic validation** of generated conversations.                                                                                                                                                                                                                                                                                                       |
| **`conversation_evaluator/`**  | Evaluation scripts for **RAG performance metrics**, including RAGAS-based assessments.                                                                                                                                                                                                                                                                              |
| **`Models/`**                  | Contains model configuration and integration scripts:<br>• `chat_gpt.py` – OpenAI GPT integration<br>• `gemini.py` – Google Gemini integration                                                                                                                                                                                                                      |
| **`rag_to_be_tested/`**        | Implementation of the target RAG architectures: System A (Naive Baseline) and System B (Agentic Retrieval). Start the RAG FastAPI service (`main.py`)                                                                                                                                                                                                               |
| **`industrial_use_case/`**     | Code for the industrial use case experiments, including Conversation Generator (CG), Conversation Validator (CV), and evaluation scripts inside for SQuAD evalaution `RAG_Evaluation_Standard/`. Data is stored in `single-hop-RAG-dataset/`.                                                                                                                       |
| **`data_preprocessing/`**      | Pipeline for news article acquisition and processing. Contains a cronjob for fetching news articles ( `guardian_fetcher`) and a Jupyter notebook to initialise `v_eval`. |
| **`streamlit_viewer/`**        | Interactive tool (`app.py`) for viewing generated conversations. |
---

## Getting Started

### Prerequisites

- **Docker**
- **Python 3.12+**
- **Poetry** (for main environment dependency management)

---

### 1. Database Setup:

1. **Start the MongoDB instance**:

Target System B requires a MongoDB instance running locally. Deploy it via Docker: 

```
docker pull mongo:latest
```

2. **Download Datasets and Vector Databases**:

The pre-embedded ChromaDBs and the raw JSON documents are hosted externally.

- ChromaDBs (`v_eval`, `v_base`, `v_advanced`): Download them from [INSERT LINK HERE].
- Raw Guardian Articles: Download the raw dataset from [INSERT LINK HERE] and import it into your running MongoDB instance.

3. **Import the raw guardian news articles into MongoDB**:

`TODO`

### 2. Environment Setup

1. **Configure API keys**
    
    Copy the environment file template and add your API credentials:
    ```bash
    cp .env.example .env
    ````
    edit `.env`
    ```bash
    GOOGLE_API_KEY=your_google_api_key
    OPENAI_API_KEY=your_openai_api_key
    MONGO_URI=mongodb://localhost:27017/
    ```
2. **Initialize the Main Environment (Poetry)**:
    The main framework (Generator, Validator, Evaluator, Context Discoverer) is managed via Poetry.
    ```bash
    poetry env use python3.12
    poetry install
    ```
3. **Initialise Sub-Environments**:
    
    The target RAG system and the Streamlit viewer run in isolated environments to prevent dependency conflicts.
    
    Setup for `rag_to_be_tested`:
        
    ```bash
    cd rag_to_be_tested
    python3.12 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    deactivate
    cd ..
    ```

    Setup for `streamlit_viewer`:
    
    ```bash
    cd streamlit_viewer
    python3.12 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    deactivate
    cd ..
    ```

---

### 3. Start the RAG Evaluation

1. **Launch the RAG FastAPI application**

   ```bash
   cd rag_to_be_tested/
   source venv/bin/activate
   uvicorn main:app --reload
   ```

2. **Start the Conversation Generator in a new terminal**

   ```bash
   poetry shell
   python3 conversation_generator/conversation_generation.py
   ```
3. **Monitor Live Generations (Optional)**:

    You can view the generated conversations in real-time using the Streamlit app:
    ```bash
    cd streamlit_viewer
    source venv/bin/activate
    streamlit run app.py
    ```
4. **After finishing, run the evaluation scripts**

   ```bash
   poetry shell
   cd conversation_evaluator
   python3 single_turn_evaluation.py
   python3 multi_turn_evaluation.py
   ```
---

## Components Overview

### Context Discoverer ([`context_discoverer/`](context_discoverer/))

Identifies latent thematic bridges between isolated document chunks.

* Executes a two-stage pipeline using filtered KNN search in the semantic vector space.
* Evaluates semantic candidates via LLM to establish valid, multi-source foundations for complex multi-hop queries.

---

### Conversation Generator ([`conversation_generator/`](conversation_generator/))

Generates **synthetic conversations** for RAG evaluation.

* Supports multiple models (GPT, Gemini).
* Role-based generation (e.g., “precise/expert”, “confused” personas).
* Implements deterministic context expansion to simulate naturally broadening dialogues.

---

### Conversation Validator ([`conversation_validator/`](conversation_validator/))

Automatically checks generated conversations. Ensures logical coherence, factual grounding, and query type alignment before inputs are passed to the target RAG system.

---

### Conversation Evaluator ([`conversation_evaluator/`](conversation_evaluator/))

Implements **RAGAS-based evaluation** and custom metrics:

* *Single-turn*: correctness, faithfulness, context precision, context recall. The metrics are evaluated separately for **single-hop** and **multi-hop** queries to isolate cross-document reasoning performance.
* *Multi-turn*: forgetfulness, context retention.

---

### Models ([`Models/`](Models/))

Integrations for large language models:

* **`chat_gpt.py`** – OpenAI GPT configuration
* **`gemini.py`** – Google Gemini API configuration

---

### Target RAG Systems ([`rag_to_be_tested/`](rag_to_be_tested/))

Implements the target RAG architectures evaluated in the thesis.

* **`main.py`** – FastAPI application serving QA endpoints.
* **`qa_chains/`** – Directory containing the RAG implementations operating on ChromaDB:
  * `qa_chain_baseline.py`: **System A (Naive Baseline)** using restrictive chunking and standard dense-vector retrieval.
  * `qa_chain_advanced.py`: **System B (Agentic Retrieval)** using an iterative LLM reasoning loop and parent-document retrieval strategies.

---

### Industrial Use Case ([`industrial_use_case/`](industrial_use_case/))

*[TODO: Add description and instructions for the industrial use case experiments, SQuAD evaluation, and data integration]*

---

## Metrics Summary

| Metric Type     | Metrics                                                      |
| --------------- | ------------------------------------------------------------ |
| **Single-Turn** | Correctness, Faithfulness, Context Precision, Context Recall |
| **Multi-Turn**  | Forgetfulness, Context Retention                             |
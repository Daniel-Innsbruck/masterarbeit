# Generated with Claude Opus 4.6, reviewed by Daniel Hillebrand

import os
import json
import time
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import List

# Load .env file and override existing environment variables
load_dotenv(override=True)
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    collection_name="target_advanced",
    persist_directory="../chroma/chroma_db_advanced",
    embedding_function=embeddings
)

# MongoDB connection for parent document lookup
mongo_client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
db = mongo_client.guardian_db

# In-memory conversation store: thread_id -> list of BaseMessage
conversation_store: dict[str, List[BaseMessage]] = {}


def llm_invoke_with_retry(llm, messages, max_retries=3, wait_time=60):
    """Invoke LLM with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if '429' in str(e):
                if attempt < max_retries - 1:
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries exceeded. Rate limit still active.")
                    raise e
            elif '503' in str(e):
                if attempt < max_retries - 1:
                    print(f"Service Unavailable. Waiting for {wait_time} seconds before retrying")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries exceeded. Service still unavailable.")
                    raise e
            else:
                raise e
    return None


def retrieve_full_document(query_text: str, n_results: int = 1) -> dict:
    """
    Parent Document Retrieval:
    1. Embed query and search ChromaDB for closest chunk
    2. Extract article_id from chunk metadata
    3. Fetch full article from MongoDB
    """
    try:
        results = vector_store.similarity_search_with_score(query_text, k=n_results)

        if not results:
            return {"status": "no_results", "content": "No matching chunks found in vector store."}

        doc, score = results[0]
        article_id = doc.metadata.get("article_id", None)
        chunk_text = doc.page_content

        if not article_id:
            return {"status": "partial", "content": chunk_text, "note": "No article_id in metadata, returning chunk only."}

        # Parent lookup in MongoDB
        article = db.articles.find_one({"id": article_id})

        if not article:
            return {"status": "partial", "content": chunk_text, "note": f"Article {article_id} not found in MongoDB, returning chunk only."}

        headline = article.get('fields', {}).get('headline', article.get('webTitle', ''))
        date = article.get('webPublicationDate', '')[:10]
        body = article.get('fields', {}).get('bodyText', '')

        full_text = f"Title: {headline}\nDate: {date}\n\n{body}"

        return {
            "status": "success",
            "content": full_text,
            "article_id": article_id,
            "relevance_score": float(score)
        }

    except Exception as e:
        return {"status": "error", "content": f"Retrieval error: {str(e)}"}


AGENT_SYSTEM_PROMPT = """You are an intelligent research assistant with access to a document retrieval tool.
Your job is to answer the user's question by retrieving and analyzing relevant documents.

You MUST respond in valid JSON format with this exact structure:
{{
    "thought": "Your reasoning about whether you have enough context to answer",
    "action": "ANSWER" or "SEARCH",
    "search_query": "The search query to use (only if action is SEARCH)",
    "final_answer": "Your complete answer (only if action is ANSWER)"
}}

RULES:
- If the context is insufficient, use action "SEARCH" with a targeted search_query.
- If you have enough context, use action "ANSWER" with a comprehensive final_answer.
- Each SEARCH retrieves a full newspaper article. Formulate precise queries.
- Consider the chat history for context about previous questions.
- ONLY output valid JSON, nothing else."""


def agentic_rag(question: str, thread_id: str, max_hops: int = 3) -> dict:
    """
    Agentic RAG with Parent Document Retrieval and multi-hop reasoning.
    """

    if thread_id not in conversation_store:
        conversation_store[thread_id] = []

    chat_history = conversation_store[thread_id]
    context_memory = []
    all_retrieved_contents = []

    for hop in range(max_hops):
        # Build context string from all retrieved documents so far
        context_str = "\n\n---\n\n".join(context_memory) if context_memory else "No documents retrieved yet."

        # Build chat history string
        history_str = ""
        if chat_history:
            for msg in chat_history:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_str += f"{role}: {msg.content}\n"

        user_message = f"""Chat History:
            {history_str if history_str else "No previous conversation."}
            
            Retrieved Context:
            {context_str}
            
            Current Question: {question}
            
            Analyze the context and decide: do you have enough information to answer, or do you need to search for more? Respond in JSON.
        """

        messages = [
            ("system", AGENT_SYSTEM_PROMPT),
            ("human", user_message)
        ]

        response = llm_invoke_with_retry(llm, messages)

        if not response:
            break

        # Parse JSON response
        try:
            response_text = response.content.strip()
            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
                response_text = response_text.rsplit("```", 1)[0]
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"[Hop {hop+1}] Failed to parse JSON, treating as final answer.")
            # Update conversation history
            conversation_store[thread_id].append(HumanMessage(content=question))
            conversation_store[thread_id].append(AIMessage(content=response.content))
            return {
                "answer": response.content,
                "context": all_retrieved_contents
            }

        thought = parsed.get("thought", "")
        action = parsed.get("action", "ANSWER")
        print(f"[Hop {hop+1}] Thought: {thought} | Action: {action}")

        if action == "SEARCH":
            search_query = parsed.get("search_query", question)
            print(f"[Hop {hop+1}] Searching: '{search_query}'")

            result = retrieve_full_document(search_query)

            if result["status"] in ("success", "partial"):
                context_memory.append(result["content"])
                all_retrieved_contents.append(result["content"])
            else:
                context_memory.append(f"Search for '{search_query}' returned no results.")

        elif action == "ANSWER":
            final_answer = parsed.get("final_answer", "I could not generate an answer.")

            # Update conversation history
            conversation_store[thread_id].append(HumanMessage(content=question))
            conversation_store[thread_id].append(AIMessage(content=final_answer))

            return {
                "answer": final_answer,
                "context": all_retrieved_contents
            }

    # Fallback: force an answer after max_hops
    print(f"[System] Max hops ({max_hops}) reached. Forcing final answer...")

    context_str = "\n\n---\n\n".join(context_memory) if context_memory else "No documents retrieved."

    fallback_message = f"""Based on ALL the context below, give the best possible answer to the question.
If some information is missing, state that clearly.

Context:
{context_str}

Question: {question}

Provide a comprehensive answer."""

    messages = [
        ("system", "You are a helpful assistant. Answer the question based on the provided context."),
        ("human", fallback_message)
    ]

    response = llm_invoke_with_retry(llm, messages)
    final_answer = response.content if response else "Could not generate an answer after maximum retrieval attempts."

    # Update conversation history
    conversation_store[thread_id].append(HumanMessage(content=question))
    conversation_store[thread_id].append(AIMessage(content=final_answer))

    return {
        "answer": final_answer,
        "context": all_retrieved_contents
    }
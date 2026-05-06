import os
import time
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  # Changed from PGVector
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict

# Load .env file and override existing environment variables
load_dotenv(override=True)
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# llm = init_chat_model("gpt-4.1-mini", model_provider="openai")
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")



embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    collection_name="guardian_chunks",
    persist_directory="../data/chroma_data",
    embedding_function=embeddings
)




prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question based on the provided context and the previous chat history. If the context does not contain the answer, state that you cannot find the answer.\n\nChat History:\n{chat_history}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be used for information retrieval. Respond only with the rephrased question."),
    ("human", "Question: {question}, chat_history: {chat_history}\n\nRephrase the question to be standalone, without context"),
])

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    messages: List[BaseMessage]


def llm_invoke_with_retry(llm, messages, max_retries=3, wait_time=60):
    """Invoke LLM with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if '429' in str(e):
                if attempt < max_retries - 1:
                    print(f"Rate limit exceeded. Waiting for 60 seconds before evaluating next batch")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries exceeded. Rate limit still active.")
                    raise e
                # try again
            elif '503' in str(e):
                if attempt < max_retries - 1:
                    print(f"Service Unavailable. Waiting for 60 seconds before evaluating next batch") 
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries exceeded. Rate limit still active.")
                    raise e
                # try again
            else:
                # For non-rate-limit errors, don't retry
                raise e
    return None


def retrieve(state: State):
    current_human_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            current_human_message = msg
            break

    current_question_content = current_human_message.content
    chat_history_for_rephrase = [msg for msg in state["messages"] if msg != current_human_message]

    if chat_history_for_rephrase:
        rephrase_messages = rephrase_prompt.invoke({
            "chat_history": chat_history_for_rephrase,
            "question": current_question_content
        })
        standalone_question_response = llm_invoke_with_retry(llm, rephrase_messages)
        standalone_question = standalone_question_response.content
    else:
        standalone_question = current_question_content

    # Retrieve k=10 from ChromaDB
    retrieved_docs = vector_store.similarity_search(standalone_question, k=10)

    return {"context": retrieved_docs, "question": current_question_content}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages_for_qa = prompt.invoke({
        "chat_history": state["messages"][:-1],
        # "chat_history": state["messages"][0],
        # "chat_history": [],
        # "chat_history": state["messages"][-4:-1],
        "context": docs_content,
        "question": state["question"]
    })
    response = llm_invoke_with_retry(llm, messages_for_qa)
    return {"answer": response.content, "messages": state["messages"] + [AIMessage(content=response.content)]}

def get_rag_graph():
    memory = MemorySaver()
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile(checkpointer=memory)
    return graph
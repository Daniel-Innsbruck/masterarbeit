from fastapi import FastAPI
from pydantic import BaseModel
from qa_chain import get_rag_graph
from langchain_core.messages import HumanMessage

app = FastAPI()
rag_graph = get_rag_graph()

class QuestionRequest(BaseModel):
    question: str
    thread_id: str

@app.post("/rag")
def ask_question(request: QuestionRequest):
    thread_id = request.thread_id
    current_checkpoint = rag_graph.get_state(config={"configurable": {"thread_id": thread_id}})

    current_messages = current_checkpoint.values.get("messages", []) if current_checkpoint and current_checkpoint.values else []
    updated_messages = current_messages + [HumanMessage(content=request.question)]

    result = rag_graph.invoke(
        {"messages": updated_messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    formatted_context = [doc.page_content for doc in result["context"]]

    # The 'result' will contain the final state after the graph execution.
    # The answer is the content of the last AIMessage in the updated messages list.
    return {"answer": result["messages"][-1].content, "context":formatted_context}

@app.get("/getThreadID")
def get_thread_id():
    """
    Returns a unique thread ID for the current conversation.
    This can be used to maintain state across multiple requests.
    """
    import uuid
    return {"thread_id": str(uuid.uuid4())}

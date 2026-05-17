from fastapi import FastAPI
from pydantic import BaseModel
from qa_chains.qa_chain import get_rag_graph
from qa_chains.qa_chain_baseline import baseline_rag
from langchain_core.messages import HumanMessage

app = FastAPI()
rag_graph = get_rag_graph()

class QuestionRequest(BaseModel):
    question: str
    thread_id: str
    mode: str = "default"  # "default" or "baseline" or "advanced"

@app.post("/rag")
def ask_question(request: QuestionRequest):
    if request.mode == "baseline":
        return baseline_rag(request.question, request.thread_id)

    if request.mode == "advanced":
        result = agentic_rag(request.question, request.thread_id)
        return {"answer": result["answer"], "context": result["context"]}

    thread_id = request.thread_id
    current_checkpoint = rag_graph.get_state(config={"configurable": {"thread_id": thread_id}})

    current_messages = current_checkpoint.values.get("messages", []) if current_checkpoint and current_checkpoint.values else []
    updated_messages = current_messages + [HumanMessage(content=request.question)]

    result = rag_graph.invoke(
        {"messages": updated_messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    formatted_context = [doc.page_content for doc in result["context"]]
    return {"answer": result["messages"][-1].content, "context": formatted_context}

@app.get("/getThreadID")
def get_thread_id():
    import uuid
    return {"thread_id": str(uuid.uuid4())}
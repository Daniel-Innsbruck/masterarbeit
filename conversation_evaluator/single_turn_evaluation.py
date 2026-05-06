from ragas.metrics import AspectCritic, SimpleCriteriaScore
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas import evaluate
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import gc
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
load_dotenv(override=True)
# print(os.environ['GOOGLE_API_KEY'])

# evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Choose the appropriate import based on your API:
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize with Google AI Studio
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-5-mini"
))

paths = [
    "../data/conversation_data_gemini-2.0-flash_turns_5_conversations_100.jsonl"
]

for path in paths:
    print(f"\nEvaluating: {path}")
    data = []
    with open(path, 'r', encoding = 'utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    list_singleturn_conversations = []
    for conversation in data:
        for turn in conversation['conversation']:
            retrieved_contexts = turn.get('context', [])
            list_singleturn_conversations.append(
                SingleTurnSample(
                    user_input=turn['question'],
                    reference=turn['answer'],  # Ground-Truth for Context Recall
                    response=turn['rag_answer'],
                    retrieved_contexts=retrieved_contexts,
                )
            )
        
    definition_correctness = "Return 1 if the AI answers the question correct; otherwise, return 0."
    correctness_aspect_critic = AspectCritic(
        name="correctness_aspect_critic",
        definition=definition_correctness,
        llm=evaluator_llm,
    )

    result = evaluate(
        dataset=EvaluationDataset(samples=list_singleturn_conversations),
        metrics=[correctness_aspect_critic,faithfulness, context_precision, context_recall],
        llm=evaluator_llm,
    )
    result_json = []
    results = result.to_pandas()
    for _, row in results.iterrows():
        user_input = row["user_input"]      
        correctness_aspect_critic = row["correctness_aspect_critic"]
        faithfulness_critic = row["faithfulness"]
        context_precision_critic = row["context_precision"]
        context_recall_critic = row["context_recall"]

    
        
        result_json.append({
            "content": user_input,
            "correctness_aspect_critic": correctness_aspect_critic,
            "faithfulness_aspect_critic": faithfulness_critic,
            "context_precision_critic": context_precision_critic,
            "context_recall_critic": context_recall_critic
        })
    avg_correctness = sum([entry["correctness_aspect_critic"] for entry in result_json]) / len(result_json)
    avg_faithfulness = sum([entry["faithfulness_aspect_critic"] for entry in result_json]) / len(result_json)
    avg_context_precision = sum([entry["context_precision_critic"] for entry in result_json]) / len(result_json)
    avg_context_recall = sum([entry["context_recall_critic"] for entry in result_json]) / len(result_json)
    final_output = {
        "file_path": path,
        "average_scores": {
            "correctness_aspect_critic": avg_correctness,
            "faithfulness_aspect_critic": avg_faithfulness,
            "context_precision_critic": avg_context_precision,
            "context_recall_critic": avg_context_recall
        },
        "detailed_results": result_json
    }
    print(f"--- RESULTS ---")
    print(
        f"Correctness: {avg_correctness:.2f} | Faithfulness: {avg_faithfulness:.2f} | Precision: {avg_context_precision:.2f} | Recall: {avg_context_recall:.2f}")

    final_path = path.replace(".jsonl", "_evaluation.json").replace("conversation_data",
                                                                    "single_turn_evaluation_results")

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    with open(final_path, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    gc.collect()
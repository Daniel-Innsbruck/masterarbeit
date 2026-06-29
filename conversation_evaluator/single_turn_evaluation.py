from ragas.metrics import AspectCritic, SimpleCriteriaScore
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas import evaluate
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import os
import gc
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
load_dotenv(find_dotenv(), override=True)
# print(os.environ['GOOGLE_API_KEY'])

# evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Choose the appropriate import based on your API:
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize with Google AI Studio
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o-mini",
))

paths = [
    "../data/2906_conversation_data_gemini-3.1-flash-lite_turns_5_conversations_1.jsonl"
]

def calculate_safe_average(result_list, key):
    valid_entries = [entry[key] for entry in result_list if entry[key] == entry[key] and entry[key] is not None]
    if len(valid_entries) == 0:
        return 0.0
    return sum(valid_entries) / len(valid_entries)


for path in paths:
    print(f"\nEvaluating: {path}")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    list_multihop_conversations = []
    list_singlehop_conversations = []

    for conversation in data:
        for turn in conversation['conversation']:
            retrieved_contexts = turn.get('context', [])
            sample = SingleTurnSample(
                user_input=turn['question'],
                reference=turn['answer'],  # Ground-Truth for Context Recall
                response=turn['rag_answer'],
                retrieved_contexts=retrieved_contexts,
            )

            if turn.get('multi_hop_flag', 0) == 1:
                list_multihop_conversations.append(sample)
            else:
                list_singlehop_conversations.append(sample)

    definition_correctness = "Return 1 if the AI answers the question correct; otherwise, return 0."
    correctness_aspect_critic = AspectCritic(
        name="correctness_aspect_critic",
        definition=definition_correctness,
        llm=evaluator_llm,
    )

    metrics_to_use = [correctness_aspect_critic, faithfulness, context_precision, context_recall]


    def run_evaluation_core(samples_list):
        if not samples_list:
            return [], {}

        result = evaluate(
            dataset=EvaluationDataset(samples=samples_list),
            metrics=metrics_to_use,
            llm=evaluator_llm,
        )

        result_json = []
        results = result.to_pandas()

        for _, row in results.iterrows():
            user_input = row["user_input"]
            correctness_val = row["correctness_aspect_critic"]
            faithfulness_val = row["faithfulness"]
            context_precision_val = row["context_precision"]
            context_recall_val = row["context_recall"]

            result_json.append({
                "content": user_input,
                "correctness_aspect_critic": correctness_val,
                "faithfulness_aspect_critic": faithfulness_val,
                "context_precision_critic": context_precision_val,
                "context_recall_critic": context_recall_val
            })

        avg_correctness = calculate_safe_average(result_json, "correctness_aspect_critic")
        avg_faithfulness = calculate_safe_average(result_json, "faithfulness_aspect_critic")
        avg_context_precision = calculate_safe_average(result_json, "context_precision_critic")
        avg_context_recall = calculate_safe_average(result_json, "context_recall_critic")

        average_scores = {
            "correctness_aspect_critic": avg_correctness,
            "faithfulness_aspect_critic": avg_faithfulness,
            "context_precision_critic": avg_context_precision,
            "context_recall_critic": avg_context_recall
        }

        return result_json, average_scores


    print(f"--- Running MULTI-HOP Evaluation ({len(list_multihop_conversations)} samples) ---")
    multi_results, multi_averages = run_evaluation_core(list_multihop_conversations)

    print(f"--- Running SINGLE-HOP Evaluation ({len(list_singlehop_conversations)} samples) ---")
    single_results, single_averages = run_evaluation_core(list_singlehop_conversations)

    final_output = {
        "file_path": path,
        "multi_hop_evaluation": {
            "average_scores": multi_averages,
            "detailed_results": multi_results
        },
        "single_hop_evaluation": {
            "average_scores": single_averages,
            "detailed_results": single_results
        }
    }

    print(f"\n--- RESULTS ---")
    if multi_averages:
        print(
            f"[MULTI-HOP]  Correctness: {multi_averages['correctness_aspect_critic']:.2f} | Faithfulness: {multi_averages['faithfulness_aspect_critic']:.2f} | Precision: {multi_averages['context_precision_critic']:.2f} | Recall: {multi_averages['context_recall_critic']:.2f}")
    if single_averages:
        print(
            f"[SINGLE-HOP] Correctness: {single_averages['correctness_aspect_critic']:.2f} | Faithfulness: {single_averages['faithfulness_aspect_critic']:.2f} | Precision: {single_averages['context_precision_critic']:.2f} | Recall: {single_averages['context_recall_critic']:.2f}")

    final_path = path.replace(".jsonl", "_evaluation.json").replace("conversation_data",
                                                                    "single_turn_evaluation_results")

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    with open(final_path, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    gc.collect()
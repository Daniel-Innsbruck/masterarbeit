import json
import os
import gc
from dotenv import load_dotenv, find_dotenv

from ragas.metrics import AspectCritic
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from ragas.messages import HumanMessage, AIMessage
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv(), override=True)

# 1. We use GPT-5-mini as the Judge to avoid Self-Enhancement Bias!
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-5-mini"
))

# INSERT THE PATH TO YOUR GENERATED .jsonl FILE HERE
paths = [
    "../data/conversation_data_gemini-2.0-flash_turns_5_conversations_100.jsonl"
]

for path in paths:
    print(f"\nEvaluating Multi-Turn for: {path}")

    # 2. Read the JSONL file
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    list_conversations = []

    for conversation in data:
        conversationlist = conversation['conversation']
        userInput = []

        # Build the chat history for the MultiTurnSample
        for turn in conversationlist:
            userInput.append(HumanMessage(content=turn['rag_input']))
            userInput.append(AIMessage(content=turn['rag_answer']))

        sample_conversation = MultiTurnSample(
            user_input=userInput
        )
        list_conversations.append(sample_conversation)

    # Metric Definitions
    definition_forgetfulness = "Return 1 if the AI forgets relevant information from earlier in the conversation, showing loss of context or failure to follow up on prior turns; otherwise, return 0."
    forgetfulness_aspect_critic = AspectCritic(
        name="forgetfulness_aspect_critic",
        definition=definition_forgetfulness,
        llm=evaluator_llm,
    )

    definition_contextretention = "Return 1 if the AI clearly retains relevant information from earlier in the conversation, demonstrating strong understanding and continuity across turns; otherwise, return 0."
    context_retention_aspect_critic = AspectCritic(
        name="context_retention_aspect_critic",
        definition=definition_contextretention,
        llm=evaluator_llm,
    )

    # 3. Start Ragas Evaluation (IMPORTANT: pass llm=evaluator_llm!)
    print(f"Starting Ragas Evaluation for {len(list_conversations)} conversations...")
    result = evaluate(
        dataset=EvaluationDataset(samples=list_conversations),
        metrics=[forgetfulness_aspect_critic, context_retention_aspect_critic],
        llm=evaluator_llm
    )

    result_json = []
    results_df = result.to_pandas()

    for _, row in results_df.iterrows():
        user_input_messages = row["user_input"]
        forgetfulness_critic = row.get("forgetfulness_aspect_critic", 0)
        context_critic = row.get("context_retention_aspect_critic", 0)

        conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in user_input_messages])

        result_json.append({
            "conversation_log": conversation_text,
            "forgetfulness_aspect_critic": forgetfulness_critic,
            "context_retention_aspect_critic": context_critic
        })

    avg_forgetfulness = sum([entry["forgetfulness_aspect_critic"] for entry in result_json]) / len(result_json)
    avg_context_retention = sum([entry["context_retention_aspect_critic"] for entry in result_json]) / len(result_json)

    final_output = {
        "file_path": path,
        "average_forgetfulness_aspect_critic": avg_forgetfulness,
        "average_context_retention_aspect_critic": avg_context_retention,
        "detailed_results": result_json
    }

    print(f"--- RESULTS ---")
    print(f"Forgetfulness: {avg_forgetfulness:.2f} | Context Retention: {avg_context_retention:.2f}")

    final_path = path.replace(".jsonl", "_evaluation.json").replace("conversation_data",
                                                                    "multi_turn_evaluation_results")

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    with open(final_path, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    gc.collect()
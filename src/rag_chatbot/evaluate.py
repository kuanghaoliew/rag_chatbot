"""
Retrieval Evaluation with Langfuse.

Evaluates retrieval quality using:
1. Hit Rate (does the correct chunk appear in top-k?)
2. MRR (Mean Reciprocal Rank)
3. LLM-as-judge (is the retrieved context sufficient to answer?)

Usage:
    uv run python -m rag_chatbot.evaluate
"""
import json
from pathlib import Path
from langfuse import get_client
from openai import OpenAI

from rag_chatbot.config import OPENAI_API_KEY, LLM_MODEL
from rag_chatbot.retriever import retrieve

openai_client = OpenAI(api_key=OPENAI_API_KEY)

langfuse = get_client()

# ── Load evaluation dataset ─────────────────────────────────────
DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_dataset.json"


def load_eval_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """Load golden dataset from JSON file."""
    if not path.exists():
        print(f"⚠ Golden dataset not found at {path}")
        print("  Create data/golden_dataset.json with your Q&A pairs.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} evaluation items from {path.name}")
    return dataset


# ── LLM-as-Judge: Context Sufficiency ──────────────────────────
def judge_context_sufficiency(question: str, context: str) -> dict:
    """
    Use LLM to judge whether the retrieved context is sufficient
    to answer the question.

    Returns: {score: 0-1, reasoning: str}
    """
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. Given a question and retrieved context, "
                    "determine if the context contains enough information to answer the question.\n"
                    "Respond with JSON: {\"score\": <0.0 to 1.0>, \"reasoning\": \"<explanation>\"}\n"
                    "- 1.0 = context fully answers the question\n"
                    "- 0.5 = context partially answers the question\n"
                    "- 0.0 = context is completely irrelevant"
                )
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved Context:\n{context}"
            }
        ],
        temperature=0,
        max_tokens=200,
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        return {"score": 0.0, "reasoning": "Failed to parse judge response"}


def judge_answer_correctness(question: str, answer: str, ground_truth: str) -> dict:
    """
    Use LLM to judge whether the generated answer matches ground truth.
    """
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. Compare the generated answer against "
                    "the ground truth answer.\n"
                    "Respond with JSON: {\"score\": <0.0 to 1.0>, \"reasoning\": \"<explanation>\"}\n"
                    "- 1.0 = answer is correct and complete\n"
                    "- 0.5 = answer is partially correct\n"
                    "- 0.0 = answer is wrong"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Generated Answer: {answer}\n\n"
                    f"Ground Truth: {ground_truth}"
                )
            }
        ],
        temperature=0,
        max_tokens=200,
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        return {"score": 0.0, "reasoning": "Failed to parse judge response"}


# ── Run evaluation ──────────────────────────────────────────────
def run_evaluation():
    """Run retrieval evaluation and log scores to Langfuse."""
    from rag_chatbot.rag_chain import rag_query, flush

    EVAL_DATASET = load_eval_dataset()
    if not EVAL_DATASET:
        return

    print("=" * 60)
    print("Running Retrieval Evaluation")
    print("=" * 60)

    total_hit = 0
    total_mrr = 0.0
    total_sufficiency = 0.0
    total_correctness = 0.0

    for i, item in enumerate(EVAL_DATASET):
        print(f"\n[{i+1}/{len(EVAL_DATASET)}] Q: {item['question']}")

        # Run RAG
        result = rag_query(item["question"])
        chunks = result["retrieved_chunks"]

        # ── Hit Rate ──
        hit = any(
            c["metadata"]["type"] == item["expected_chunk_type"]
            for c in chunks
        )
        total_hit += int(hit)
        print(f"  Hit (type={item['expected_chunk_type']}): {'✓' if hit else '✗'}")

        # ── MRR ──
        mrr = 0.0
        for rank, c in enumerate(chunks, 1):
            if c["metadata"]["type"] == item["expected_chunk_type"]:
                mrr = 1.0 / rank
                break
        total_mrr += mrr
        print(f"  MRR: {mrr:.2f}")

        # ── LLM Judge: Context Sufficiency ──
        sufficiency = judge_context_sufficiency(
            item["question"], result["context"]
        )
        total_sufficiency += sufficiency["score"]
        print(f"  Context Sufficiency: {sufficiency['score']:.1f} - {sufficiency['reasoning']}")

        # ── LLM Judge: Answer Correctness ──
        correctness = judge_answer_correctness(
            item["question"], result["answer"], item["ground_truth"]
        )
        total_correctness += correctness["score"]
        print(f"  Answer Correctness: {correctness['score']:.1f} - {correctness['reasoning']}")

        # ── Log scores to Langfuse ──
        with langfuse.start_as_current_observation(as_type="span", name="evaluation") as span:
            span.update(input=item["question"], output=result["answer"],
                        metadata={"ground_truth": item["ground_truth"]})
            trace_id = langfuse.get_current_trace_id()

        langfuse.create_score(name="hit_rate", value=float(int(hit)), trace_id=trace_id, data_type="NUMERIC")
        langfuse.create_score(name="mrr", value=float(mrr), trace_id=trace_id, data_type="NUMERIC")
        langfuse.create_score(name="context_sufficiency", value=float(sufficiency["score"]),
                            trace_id=trace_id, data_type="NUMERIC", comment=sufficiency["reasoning"])
        langfuse.create_score(name="answer_correctness", value=float(correctness["score"]),
                            trace_id=trace_id, data_type="NUMERIC", comment=correctness["reasoning"])

    # ── Summary ──
    n = len(EVAL_DATASET)
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  Hit Rate:              {total_hit/n:.2%}")
    print(f"  Mean Reciprocal Rank:  {total_mrr/n:.3f}")
    print(f"  Avg Sufficiency:       {total_sufficiency/n:.2f}")
    print(f"  Avg Correctness:       {total_correctness/n:.2f}")
    print("=" * 60)
    print("📊 View detailed traces in Langfuse: http://localhost:3000")

    flush()
    langfuse.flush()


def main():
    """CLI entry point."""
    run_evaluation()


if __name__ == "__main__":
    main()

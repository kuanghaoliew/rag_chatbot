"""
RAG Chain with Langfuse v3 observability.

Uses OpenAI native message format:
  - System message: instruction from prompt.yaml
  - Previous messages: last N conversation turns (user/assistant)
  - Current message: retrieved documents + user query

Every query is traced end-to-end:
  Trace
  ├── Span: retrieval
  ├── Generation: LLM call
  └── metadata (latency, token usage, retrieved chunks)
"""
import yaml
from langfuse import observe, get_client
from openai import OpenAI

from rag_chatbot.config import OPENAI_API_KEY, LLM_MODEL, TOP_K, PROMPT_PATH
from rag_chatbot.retriever import retrieve, format_context

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── Load prompt template ─────────────────────────────────────────

def load_prompt_template() -> dict:
    """Load prompt config from YAML file."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PROMPT_CONFIG = load_prompt_template()


# ── Format prompt sections ───────────────────────────────────────

def get_recent_history(chat_history: list[dict] | None) -> list[dict]:
    """
    Get the last N turns from chat history as message objects.
    Returns messages in OpenAI format: [{"role": "user", "content": "..."}, ...]
    """
    if not chat_history:
        return []

    max_turns = PROMPT_CONFIG.get("max_history_turns", 3)
    # Each turn is 2 messages (user + assistant), take last N turns
    max_messages = max_turns * 2
    return chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history


def format_retrieved_documents(chunks: list[dict]) -> str:
    """
    Format chunks as numbered documents:
      [Doc 1] (Page 3, text) content...
      [Doc 2] (Page 5, table) content...
    """
    if not chunks:
        return "(No documents retrieved)"

    docs = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        page = meta.get("page", "?")
        chunk_type = meta.get("type", "text")
        text = chunk["text"].strip()
        docs.append(f"[Doc {i}] (Page {page}, {chunk_type})\n{text}")

    return "\n\n".join(docs)


# ── LLM call ─────────────────────────────────────────────────────

@observe(as_type="generation")
def llm_generate(messages: list[dict], model: str = LLM_MODEL) -> str:
    """Call OpenAI with Langfuse generation tracking."""
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    client = get_client()
    client.update_current_generation(
        model=model,
        usage_details={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
        metadata={
            "finish_reason": response.choices[0].finish_reason,
        }
    )

    return answer


# ── Retrieval ─────────────────────────────────────────────────────

@observe()
def retrieve_with_tracing(query: str, top_k: int = TOP_K) -> tuple[list[dict], str]:
    """Retrieve relevant chunks with Langfuse span tracking."""
    chunks = retrieve(query, top_k=top_k)
    context = format_context(chunks)

    client = get_client()
    client.update_current_span(
        metadata={
            "query": query,
            "top_k": top_k,
            "num_retrieved": len(chunks),
            "chunk_types": [c["metadata"]["type"] for c in chunks],
            "distances": [c["distance"] for c in chunks],
            "pages": [c["metadata"].get("page") for c in chunks],
        }
    )

    return chunks, context


# ── RAG pipeline ──────────────────────────────────────────────────

@observe()
def rag_query(user_query: str, chat_history: list[dict] = None) -> dict:
    """
    Full RAG pipeline: retrieve → build messages → generate.

    Args:
        user_query: The current question
        chat_history: List of {"role": "user"/"assistant", "content": "..."} dicts

    Returns dict with: answer, retrieved_chunks, context
    """
    # 1. Retrieve
    chunks, context = retrieve_with_tracing(user_query)

    # 2. Build messages in native OpenAI format
    messages = [
        {"role": "system", "content": PROMPT_CONFIG["system_instruction"].strip()}
    ]

    # 3. Add recent conversation history
    recent_history = get_recent_history(chat_history)
    messages.extend(recent_history)

    # 4. Add current query with retrieved documents
    retrieved_docs = format_retrieved_documents(chunks)
    current_message = f"""# Retrieved Documents
{retrieved_docs}

# Question
{user_query}"""

    messages.append({"role": "user", "content": current_message})

    # 5. Generate
    answer = llm_generate(messages)

    # 6. Update trace
    client = get_client()
    client.update_current_trace(
        input=user_query,
        output=answer,
        metadata={
            "num_chunks": len(chunks),
            "chunk_types": [c["metadata"]["type"] for c in chunks],
            "has_history": chat_history is not None and len(chat_history) > 0,
        }
    )

    return {
        "answer": answer,
        "retrieved_chunks": chunks,
        "context": context,
    }


def flush():
    """Ensure all Langfuse events are sent."""
    get_client().flush()

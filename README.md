# RAG Chatbot over PDF

A production-ready conversational RAG (Retrieval-Augmented Generation) chatbot for querying PDF documents, powered by OpenAI GPT-4.1, ChromaDB, and Langfuse observability.

## Features

### PDF Processing
- **Multi-modal extraction** with pdfplumber + GPT-4.1 vision
  - Text with automatic header detection (numbered sections + font size analysis)
  - Tables converted to clean markdown via vision API
  - Images described with GPT-4.1 vision
  - Footnote detection and filtering
- **Smart chunking**
  - Section-aware splitting with hierarchy preservation
  - Cross-page text merging (no artificial paragraph breaks)
  - Separate handling for text, tables, and images
  - Each chunk prefixed with full section path (e.g., `# 1 Intro > ## 1.1 What is AI`)

### Conversational RAG
- **Multi-turn conversations** with native OpenAI message format
  - Maintains last 3 conversation turns as proper message objects
  - Better context understanding than text-based history
- **Similarity-based retrieval** with ChromaDB (cosine distance)
  - Configurable top-k and similarity threshold
  - Returns text, tables, and image descriptions
- **End-to-end tracing** with Langfuse v3
  - Retrieval spans with metadata (chunk types, distances, pages)
  - LLM generation tracking (tokens, latency, finish reason)
  - Full trace lineage from query to answer

### Evaluation Framework
- **Retrieval metrics**
  - Hit Rate (does the correct chunk appear in top-k?)
  - Mean Reciprocal Rank (MRR)
- **LLM-as-judge** scoring
  - Context sufficiency (is retrieved context enough to answer?)
  - Answer correctness (does answer match ground truth?)
- **Langfuse integration** - all scores logged for analysis

### Streamlit UI
- Interactive chat interface with message history
- Retrieved sources viewer (expandable, with page numbers and distances)
- Real-time streaming responses
- Clean, responsive layout

## Tech Stack

- **PDF Parsing**: pdfplumber (MIT license, production-safe)
- **LLM**: OpenAI GPT-4.1 (text + vision)
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector DB**: ChromaDB (persistent, cosine similarity)
- **Observability**: Langfuse v3 (self-hosted or cloud)
- **UI**: Streamlit
- **Package Manager**: uv

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key
- (Optional) Langfuse instance (local Docker or cloud)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd rag-chatbot
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Configure environment variables**
   Create a `.env` file:
   ```bash
   # OpenAI
   OPENAI_API_KEY=sk-...

   # Langfuse (optional - for observability)
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=http://localhost:3000
   ```

4. **Start Langfuse (optional)**
   ```bash
   # Using Docker
   docker-compose up -d
   # Or use Langfuse Cloud
   ```

## Usage

### 1. Ingest a PDF

Convert a PDF to markdown, chunk it, embed it, and store in ChromaDB:

```bash
uv run ingest --pdf data/your-document.pdf
```

**Options:**
- `--pdf`: Path to PDF file (required)
- `--output-dir`: Directory to save parsed markdown (default: `parsed_output`)
- `--parse-only`: Only convert PDF to markdown without embedding

**Example:**
```bash
# Full ingestion pipeline
uv run ingest --pdf data/mgf-for-agentic-ai.pdf

# Parse only (no embedding)
uv run ingest --pdf data/mgf-for-agentic-ai.pdf --parse-only
```

**What it does:**
1. Parses PDF with header detection, table extraction, and image description
2. Saves markdown to `parsed_output/<filename>.md`
3. Chunks the markdown (section-aware, cross-page merging)
4. Generates embeddings (OpenAI text-embedding-3-small)
5. Stores in ChromaDB (`chroma_db/`)

### 2. Run the Chatbot

Start the Streamlit UI:

```bash
uv run streamlit run app.py
```

Open http://localhost:8501 in your browser and start chatting!

**Features:**
- Ask questions about your ingested PDF
- View retrieved sources (with page numbers, chunk types, and similarity scores)
- Multi-turn conversations with context awareness
- All interactions traced in Langfuse

### 3. Evaluate Performance

Run evaluation on a golden dataset:

```bash
uv run evaluate
```

**Requirements:**
Create `data/golden_dataset.json` with your evaluation questions:

```json
[
  {
    "question": "What is agentic AI?",
    "ground_truth": "Agentic AI refers to AI systems that can act autonomously...",
    "expected_chunk_type": "text"
  },
  {
    "question": "What are the key metrics?",
    "ground_truth": "The key metrics are precision, recall, and F1 score.",
    "expected_chunk_type": "table"
  }
]
```

**Metrics computed:**
- Hit Rate: % of queries where the expected chunk type appears in top-k
- MRR: Mean reciprocal rank of the expected chunk
- Context Sufficiency: LLM judge score (0-1) on whether context is sufficient
- Answer Correctness: LLM judge score (0-1) on answer quality

**View results:**
- Console output shows per-query scores
- Langfuse dashboard shows detailed traces and aggregate metrics

## Project Structure

```
rag-chatbot/
├── app.py                          # Streamlit UI
├── pyproject.toml                  # Dependencies and scripts
├── .env                            # Environment variables (not in git)
├── data/                           # PDF files and evaluation dataset
│   ├── golden_dataset.json
│   └── *.pdf
├── src/rag_chatbot/
│   ├── ingest.py                   # PDF → markdown → chunks → ChromaDB
│   ├── retriever.py                # Query ChromaDB for relevant chunks
│   ├── rag_chain.py                # RAG pipeline with Langfuse tracing
│   ├── evaluate.py                 # Evaluation framework
│   ├── config.py                   # Configuration (models, paths, etc.)
│   └── prompt.yaml                 # System instruction and prompt template
├── chroma_db/                      # ChromaDB persistent storage (not in git)
├── parsed_output/                  # Parsed markdown files (not in git)
└── test_message_format.py          # Test script for message format
```

## Configuration

Edit `src/rag_chatbot/config.py` or override via environment variables:

```python
# OpenAI
LLM_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Chunking
CHUNK_SIZE = 1000          # Target chunk size (chars)
CHUNK_OVERLAP = 200        # Overlap between chunks

# Retrieval
TOP_K = 5                  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.4 # Minimum similarity score (0-1)

# Langfuse
LANGFUSE_BASE_URL = "http://localhost:3000"
```

## Prompt Template

The system instruction and prompt structure are defined in `src/rag_chatbot/prompt.yaml`:

```yaml
system_instruction: |
  You are a helpful assistant that answers questions based on the provided context from a PDF document.

  Rules:
  - Answer ONLY based on the retrieved documents provided below
  - If the documents don't contain enough information, say "I don't have enough information in the document to answer this"
  - Cite the document reference (e.g. [Doc 1]) and page number when possible

max_history_turns: 3  # Number of conversation turns to include
```

## How It Works

### PDF Ingestion Pipeline

1. **PDF → Markdown**
   - Detect body font size across the PDF
   - Skip cover and table-of-contents pages
   - Extract text with header detection:
     - Numbered sections: `1`, `1.1`, `1.1.1` → `#`, `##`, `###`
     - Font size analysis: larger-than-body text → headers
     - Footnotes: smaller-than-body text → excluded from headers
   - Extract tables → crop as images → GPT-4.1 vision → markdown tables
   - Extract images → crop → GPT-4.1 vision → text descriptions

2. **Markdown → Chunks**
   - Merge all text across pages (no artificial breaks)
   - Split at section headers (`#`, `##`, `###`)
   - Tables and images → isolated chunks
   - Each chunk prefixed with section hierarchy
   - Oversized sections → split by paragraphs, then sentences

3. **Chunks → Embeddings → ChromaDB**
   - Generate embeddings (text-embedding-3-small)
   - Store in ChromaDB with metadata (page, type, etc.)
   - Persistent storage in `chroma_db/`

### RAG Query Pipeline

1. **User query** → embed with same model
2. **ChromaDB retrieval** → top-k chunks (cosine similarity)
3. **Filter** by similarity threshold (0.4 default)
4. **Build messages** in native OpenAI format:
   ```python
   [
     {"role": "system", "content": "system instruction"},
     {"role": "user", "content": "previous question"},
     {"role": "assistant", "content": "previous answer"},
     # ... last 3 turns ...
     {"role": "user", "content": "retrieved docs + current question"}
   ]
   ```
5. **LLM generation** → GPT-4.1
6. **Langfuse tracing** → log retrieval + generation + metadata

### Observability

Every query is traced end-to-end in Langfuse:

```
Trace (rag_query)
├── Span: retrieve_with_tracing
│   └── metadata: {query, top_k, num_retrieved, chunk_types, distances, pages}
├── Generation: llm_generate
│   └── usage: {input_tokens, output_tokens, total_tokens}
│   └── metadata: {finish_reason}
└── metadata: {num_chunks, chunk_types, has_history}
```

View traces at http://localhost:3000 (or your Langfuse URL)

## Development

### Running Tests

Test the native message format:
```bash
uv run python test_message_format.py
```

### Adding Custom Scripts

Define in `pyproject.toml`:
```toml
[project.scripts]
ingest = "rag_chatbot.ingest:main"
evaluate = "rag_chatbot.evaluate:main"
```

Run with:
```bash
uv run <script-name>
```

## Troubleshooting

### ChromaDB not found
```
Error: Collection 'pdf_rag' not found
```
→ Run `uv run ingest --pdf <your-pdf>` first to create the collection

### Langfuse connection failed
```
Error: Failed to connect to Langfuse
```
→ Check that Langfuse is running and `LANGFUSE_BASE_URL` is correct
→ Or comment out Langfuse imports to run without observability

### GPT-4.1 model not found
```
Error: The model 'gpt-4.1' does not exist
```
→ Update `LLM_MODEL` in `config.py` to a valid model (e.g., `gpt-4o`)

### Vision API errors
```
Error: Table/image vision failed
```
→ Check that your OpenAI API key has access to vision models
→ Vision errors are non-fatal; ingestion will continue without those items

## License

MIT License (pdfplumber, core libraries)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- [pdfplumber](https://github.com/jsvine/pdfplumber) for robust PDF parsing
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Langfuse](https://langfuse.com/) for observability
- OpenAI for LLM and embedding models

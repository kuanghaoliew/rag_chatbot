"""
Streamlit Chatbot UI for RAG over PDF.

Run: uv run streamlit run app.py
"""
import streamlit as st
from rag_chatbot.rag_chain import rag_query, flush

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Chatbot")
st.caption("Ask questions about the ingested PDF document — powered by pdfplumber + ChromaDB + GPT-4.1")

# ── Sidebar ──
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Pipeline:**
    1. PDF parsed with **pdfplumber** (text, tables, images)
    2. Chunks embedded with **text-embedding-3-small**
    3. Stored in **ChromaDB**
    4. Retrieved & answered by **GPT-4.1**
    5. Traced in **Langfuse**
    """)
    st.markdown("---")
    st.markdown("[🔗 Open Langfuse](http://localhost:3000)")
    show_sources = st.checkbox("Show retrieved sources", value=True)

# ── Chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and show_sources:
            with st.expander("📚 Retrieved Sources"):
                for i, src in enumerate(msg["sources"]):
                    meta = src["metadata"]
                    st.markdown(
                        f"**Chunk {i+1}** | Type: `{meta.get('type')}` | "
                        f"Page: `{meta.get('page', '?')}` | "
                        f"Distance: `{src['distance']:.4f}`"
                    )
                    st.code(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""),
                            language=None)

# ── Chat input ──
if prompt := st.chat_input("Ask a question about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                # Build clean history (role + content only, no sources metadata)
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                result = rag_query(prompt, chat_history=chat_history)
                answer = result["answer"]
                chunks = result["retrieved_chunks"]
                flush()

                st.markdown(answer)

                if show_sources:
                    with st.expander("📚 Retrieved Sources"):
                        for i, src in enumerate(chunks):
                            meta = src["metadata"]
                            st.markdown(
                                f"**Chunk {i+1}** | Type: `{meta.get('type')}` | "
                                f"Page: `{meta.get('page', '?')}` | "
                                f"Distance: `{src['distance']:.4f}`"
                            )
                            st.code(
                                src["text"][:300] + ("..." if len(src["text"]) > 300 else ""),
                                language=None,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": chunks,
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

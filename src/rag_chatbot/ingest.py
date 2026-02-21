"""
PDF Ingestion Pipeline (pdfplumber + GPT-4.1 Vision)

Flow: PDF → Markdown (headers via numbered sections + font size)
      → save locally → chunk (cross-page, section-aware) → embed → ChromaDB

Tables and images are both processed via GPT-4.1 vision for maximum accuracy.
pdfplumber handles text extraction and provides bounding boxes for tables/images.

License: pdfplumber is MIT — safe for commercial/production use.
"""
import argparse
import base64
import io
import os
import re
import uuid
from collections import Counter
from pathlib import Path

import pdfplumber
import chromadb
from openai import OpenAI

from rag_chatbot.config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL,
)

client = OpenAI(api_key=OPENAI_API_KEY)

# Regex for numbered section headers like "1 Title", "1.1 Title", "2.1.3 Title"
NUMBERED_HEADER_RE = re.compile(
    r'^(\d+(?:\.\d+)*)\s+([A-Z].*)'
)

# Regex for TOC lines: "Some Title ....................... 12"
TOC_LINE_RE = re.compile(r'\.{4,}\s*\d+\s*$')


# ── 1. Convert PDF to Markdown ──────────────────────────────────

def _is_toc_page(page) -> bool:
    """
    Detect if a page is a Table of Contents by looking for dot leader patterns.
    TOC lines look like: "Executive Summary ........................................ 1"
    If >30% of non-empty lines match, it's a TOC page.
    """
    text = page.extract_text() or ""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False
    toc_count = sum(1 for l in lines if TOC_LINE_RE.search(l))
    return toc_count / len(lines) > 0.3


def _is_cover_page(page, page_num: int) -> bool:
    """
    Detect if page 1 is a cover page (very little body text, mostly title/logo).
    Only applies to page 1.
    """
    if page_num != 1:
        return False
    text = page.extract_text() or ""
    # Cover pages typically have very few lines of actual text
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return len(lines) < 15


def _detect_body_font_size(pdf) -> float:
    """Detect the most common font size across the PDF (= body text)."""
    all_sizes = []
    for page in pdf.pages:
        for char in page.chars:
            if char.get("text", "").strip():
                all_sizes.append(round(char["size"], 1))
    if not all_sizes:
        return 12.0
    return Counter(all_sizes).most_common(1)[0][0]


def _header_level_from_numbering(numbering: str) -> int:
    """
    Determine markdown header level from section numbering depth.
    '1' or '2'         → 1 (# h1)
    '1.1' or '2.3'     → 2 (## h2)
    '1.1.1' or '2.3.1' → 3 (### h3)
    """
    depth = numbering.count('.') + 1
    return min(depth, 3)


def _extract_text_with_headers(page, body_size: float) -> str:
    """
    Extract text using extract_text() for correct spacing,
    then detect headers via:
    1. Numbered sections (regex): "1 Title", "1.1 Title", "1.1.1 Title"
    2. Font size analysis: lines significantly larger than body text

    Footnotes (lines with font size smaller than body) are excluded from
    header detection to prevent false positives like "1 Adapted from..."
    """
    text = page.extract_text() or ""
    if not text.strip():
        return text

    # ── Build font-size maps from character analysis ──
    font_header_lines = {}   # stripped text → header level
    footnote_lines = set()   # stripped text of footnote lines

    if page.chars:
        lines_dict: dict[float, list] = {}
        for char in page.chars:
            if not char.get("text", "").strip():
                continue
            y_key = round(char["top"] / 2) * 2
            lines_dict.setdefault(y_key, []).append(char)

        for y_key in sorted(lines_dict.keys()):
            chars = lines_dict[y_key]
            line_sizes = [round(c["size"], 1) for c in chars if c["text"].strip()]
            if not line_sizes:
                continue

            raw = "".join(c["text"] for c in sorted(chars, key=lambda c: c["x0"]))
            raw_stripped = raw.strip()

            if not raw_stripped or len(raw_stripped) < 2:
                continue

            dominant = Counter(line_sizes).most_common(1)[0][0]
            ratio = dominant / body_size if body_size > 0 else 1.0

            # Smaller than body text → footnote
            if ratio < 0.95:
                footnote_lines.add(re.sub(r'\s+', ' ', raw_stripped))
            # Larger than body text → candidate header (non-numbered)
            elif len(raw_stripped) <= 120:
                normalized = re.sub(r'\s+', ' ', raw_stripped)
                if ratio >= 1.6:
                    font_header_lines[normalized] = 1
                elif ratio >= 1.3:
                    font_header_lines[normalized] = 2
                elif ratio >= 1.15:
                    font_header_lines[normalized] = 3

    # ── Apply headers to each line ──
    result_lines = []
    for line in text.split('\n'):
        stripped = line.strip()

        if not stripped:
            result_lines.append(line)
            continue

        # Already a markdown header
        if stripped.startswith('#'):
            result_lines.append(line)
            continue

        # 1. Skip footnotes — never apply header detection
        #    Uses containment check because char-level Y-grouping may split
        #    superscript footnote numbers from the footnote text body
        normalized = re.sub(r'\s+', ' ', stripped)
        if footnote_lines and any(fn in normalized for fn in footnote_lines if len(fn) > 10):
            result_lines.append(line)
            continue

        # 2. Check numbered section pattern (e.g. "1.2.1 Title")
        m = NUMBERED_HEADER_RE.match(stripped)
        if m:
            numbering = m.group(1)
            level = _header_level_from_numbering(numbering)
            prefix = "#" * level
            result_lines.append(f"{prefix} {stripped}")
            continue

        # 3. Check font-size-based headers (e.g. "Executive Summary")
        if normalized in font_header_lines:
            level = font_header_lines[normalized]
            prefix = "#" * level
            result_lines.append(f"{prefix} {stripped}")
            continue

        # 4. Regular text
        result_lines.append(line)

    return '\n'.join(result_lines)


def _crop_to_base64(page, bbox) -> str:
    """Crop a region from a page and return as base64 PNG."""
    cropped = page.crop(bbox)
    pil_img = cropped.to_image(resolution=200).original
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def describe_table_with_llm(page, bbox) -> str:
    """Crop table region and use GPT-4.1 vision to get clean markdown table."""
    try:
        img_base64 = _crop_to_base64(page, bbox)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert this table into a clean markdown table. "
                                "Preserve all data accurately. Handle merged cells by "
                                "placing content in the correct row and column. "
                                "Return ONLY the markdown table, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            }],
            max_tokens=1000,
        )
        result = response.choices[0].message.content.strip()
        result = re.sub(r'^```(?:markdown)?\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        return result.strip()
    except Exception as e:
        print(f"   ⚠ Table vision failed: {e}")
        return ""


def describe_image_with_llm(page, bbox, page_num: int) -> str:
    """Crop image region and use GPT-4.1 vision to describe it."""
    try:
        img_base64 = _crop_to_base64(page, bbox)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image from a PDF document in detail. "
                                "Include any text, data, labels, or key information visible. "
                                "If it's a chart or graph, describe the data trends. "
                                "Be factual and concise."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"   ⚠ Image description failed on page {page_num}: {e}")
        return "[Image could not be described]"


def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert entire PDF to a single Markdown string.

    - Text: extracted with header detection (numbered sections + font size)
    - Footnotes: detected via smaller-than-body font size, left as plain text
    - Tables: cropped as images → GPT-4.1 vision → clean markdown tables
    - Images: cropped → GPT-4.1 vision → text descriptions as blockquotes
    """
    print(f"[1/5] Parsing PDF: {pdf_path}")
    md_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        total_tables = 0
        total_images = 0

        body_size = _detect_body_font_size(pdf)
        print(f"   → Detected body font size: {body_size}pt")

        for page_num, page in enumerate(pdf.pages, 1):
            # ── Skip non-content pages ──
            if _is_cover_page(page, page_num):
                print(f"   → Skipping page {page_num} (cover page)")
                continue
            if _is_toc_page(page):
                print(f"   → Skipping page {page_num} (table of contents)")
                continue

            md_parts.append(f"---\n<!-- PAGE {page_num} -->")

            # ── Detect table regions ──
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            # ── Extract text (excluding table regions) with header detection ──
            if table_bboxes:
                filtered_page = page
                for bbox in table_bboxes:
                    filtered_page = filtered_page.outside_bbox(bbox)
                text = _extract_text_with_headers(filtered_page, body_size)
            else:
                text = _extract_text_with_headers(page, body_size)

            text = text.strip()
            if text:
                text = re.sub(r'\n\s*\n+', '\n\n', text)
                md_parts.append(text)

            # ── Tables via vision ──
            for table in tables:
                total_tables += 1
                print(f"   → Table {total_tables} (page {page_num}) → vision...")
                md_table = describe_table_with_llm(page, table.bbox)
                if md_table:
                    md_parts.append(md_table)

            # ── Images via vision ──
            for img_info in page.images:
                try:
                    x0 = img_info["x0"]
                    y0 = img_info["top"]
                    x1 = img_info["x1"]
                    y1 = img_info["bottom"]
                    width = x1 - x0
                    height = y1 - y0

                    if width < 50 or height < 50:
                        continue

                    total_images += 1
                    print(f"   → Image {total_images} (page {page_num}) → vision...")
                    desc = describe_image_with_llm(page, (x0, y0, x1, y1), page_num)
                    md_parts.append(f"> **[Figure on page {page_num}]**: {desc}")

                except Exception as e:
                    print(f"   ⚠ Could not extract image on page {page_num}: {e}")

        total_pages = len(pdf.pages)

    print(f"   → Processed {total_pages} pages ({total_tables} tables, {total_images} images)")
    return "\n\n".join(part for part in md_parts if part.strip())


# ── 2. Save Markdown locally ────────────────────────────────────

def save_markdown(markdown: str, pdf_path: str, output_dir: str = "parsed_output") -> str:
    """Save markdown to local file. Returns the saved filepath."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = Path(pdf_path).stem
    md_path = out / f"{stem}.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"[2/5] Saved markdown → {md_path} ({len(markdown):,} chars)")
    return str(md_path)


# ── 3. Chunk the Markdown (section-based) ────────────────────────

def chunk_markdown(markdown: str, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Section-based chunking with cross-page text merging.

    Strategy:
    1. Parse pages, extract tables/images as single isolated chunks
    2. Merge ALL text across pages into one stream (no paragraph splits at
       page boundaries)
    3. Split merged text at section header boundaries (# / ## / ###)
       — each section becomes one chunk
    4. If a section exceeds chunk_size * 1.5, split by paragraphs
    5. If a paragraph still exceeds, split by sentences
    6. Every chunk is prefixed with its full section hierarchy path
       e.g. "# 1 Introduction > ## 1.1 What is Agentic AI\n\n{body}"
    """
    chunks = []

    page_pattern = r'<!-- PAGE (\d+) -->'
    parts = re.split(f'(?={page_pattern})', markdown)

    # ── Pass 1: collect text stream + table/image chunks ──
    merged_text_parts = []   # list of (text_str, page_no)
    current_sections = {}

    for part in parts:
        part = part.strip()
        if not part:
            continue

        page_match = re.match(page_pattern, part)
        page_no = int(page_match.group(1)) if page_match else 0

        content = re.sub(page_pattern, '', part).strip()
        content = content.lstrip('-').strip()
        if not content:
            continue

        segments = _split_content_segments(content)

        for seg_type, seg_text in segments:
            seg_text = seg_text.strip()
            if len(seg_text) < 20:
                continue

            _update_sections_from_text(seg_text, current_sections)

            if seg_type in ("table", "image"):
                prefix = _build_section_prefix(current_sections)
                prefixed = f"{prefix}\n\n{seg_text}" if prefix else seg_text
                chunks.append({
                    "text": prefixed,
                    "metadata": {"type": seg_type, "page": page_no},
                })
            else:
                merged_text_parts.append((seg_text, page_no))

    # ── Pass 2: merge text across pages into one stream ──
    full_text = ""
    page_map = []   # list of (char_offset, page_no)
    for text_part, page_no in merged_text_parts:
        page_map.append((len(full_text), page_no))
        if full_text:
            full_text += "\n"
        full_text += text_part

    # ── Pass 3: split merged text at section headers ──
    if full_text:
        sections = _split_text_by_sections(full_text, page_map)

        for section in sections:
            body = section["body"].strip()
            if len(body) < 20:
                continue

            prefix = section["prefix"]
            page_no = section["page"]
            max_size = int(chunk_size * 1.5)

            if len(body) <= max_size:
                # Section fits in one chunk
                prefixed = f"{prefix}\n\n{body}" if prefix else body
                chunks.append({
                    "text": prefixed,
                    "metadata": {"type": "text", "page": page_no},
                })
            else:
                # Section too large → split by paragraphs, then sentences
                sub_chunks = _split_oversized_section(body, chunk_size)
                for sub in sub_chunks:
                    if len(sub.strip()) < 20:
                        continue
                    prefixed = f"{prefix}\n\n{sub}" if prefix else sub
                    chunks.append({
                        "text": prefixed,
                        "metadata": {"type": "text", "page": page_no},
                    })

    text_count = sum(1 for c in chunks if c["metadata"]["type"] == "text")
    table_count = sum(1 for c in chunks if c["metadata"]["type"] == "table")
    image_count = sum(1 for c in chunks if c["metadata"]["type"] == "image")
    print(f"[3/5] Chunked → {len(chunks)} chunks "
          f"(text: {text_count}, tables: {table_count}, images: {image_count})")
    return chunks


def _split_text_by_sections(
    full_text: str,
    page_map: list[tuple[int, int]],
) -> list[dict]:
    """
    Split merged text stream at header boundaries (# / ## / ###).

    Returns list of:
      {"prefix": "# 1 Intro > ## 1.1 AI", "body": "section text...", "page": 3}
    """
    header_re = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)

    # Find all header positions
    header_positions = []
    for m in header_re.finditer(full_text):
        level = len(m.group(1))
        title = m.group(2).strip()
        header_positions.append((m.start(), level, title))

    if not header_positions:
        # No headers found — return entire text as one section
        page_no = _find_page_from_map(0, page_map)
        return [{"prefix": "", "body": full_text, "page": page_no}]

    sections = []
    active_sections = {}   # level → title

    for i, (start, level, title) in enumerate(header_positions):
        # Update section hierarchy
        active_sections[level] = title
        for l in list(active_sections.keys()):
            if l > level:
                del active_sections[l]

        # Section body = text from this header to the next header (or end)
        if i + 1 < len(header_positions):
            end = header_positions[i + 1][0]
        else:
            end = len(full_text)

        # Body starts after the header line itself
        header_line_end = full_text.index('\n', start) + 1 if '\n' in full_text[start:end] else end
        body = full_text[header_line_end:end].strip()

        prefix = _build_section_prefix(active_sections)
        page_no = _find_page_from_map(start, page_map)

        sections.append({"prefix": prefix, "body": body, "page": page_no})

    # Handle any text BEFORE the first header
    first_header_start = header_positions[0][0]
    if first_header_start > 0:
        preamble = full_text[:first_header_start].strip()
        if len(preamble) >= 20:
            page_no = _find_page_from_map(0, page_map)
            sections.insert(0, {"prefix": "", "body": preamble, "page": page_no})

    return sections


def _split_oversized_section(text: str, chunk_size: int) -> list[str]:
    """
    Split an oversized section body into smaller chunks.

    Strategy:
    1. Split by paragraphs (double newline)
    2. Accumulate paragraphs until chunk_size is reached
    3. If a single paragraph exceeds chunk_size, split by sentences
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Single paragraph exceeds limit → split by sentences
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            sentence_chunks = _split_by_sentences(para, chunk_size)
            chunks.extend(sentence_chunks)
            continue

        # Would adding this paragraph exceed the limit?
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _split_by_sentences(text: str, chunk_size: int) -> list[str]:
    """
    Last-resort split: break text at sentence boundaries.
    Sentences are detected by '. ', '? ', '! ' followed by uppercase or newline.
    """
    sentence_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z\d])')
    sentences = sentence_re.split(text)

    chunks = []
    current = ""

    for sent in sentences:
        if current and len(current) + len(sent) + 1 > chunk_size:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}" if current else sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _find_page_from_map(char_offset: int, page_map: list[tuple[int, int]]) -> int:
    """Find which page a character offset belongs to using the page map."""
    page_no = 0
    for boundary_offset, boundary_page in page_map:
        if char_offset >= boundary_offset:
            page_no = boundary_page
        else:
            break
    return page_no


def _update_sections_from_text(text: str, current_sections: dict):
    """Update running section state from headers in text."""
    for line in text.split('\n'):
        match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            current_sections[level] = title
            for l in list(current_sections.keys()):
                if l > level:
                    del current_sections[l]


def _build_section_prefix(current_sections: dict) -> str:
    """Build section hierarchy string: '# 1 Intro > ## 1.1 What is AI'"""
    if not current_sections:
        return ""
    parts = []
    for level in sorted(current_sections.keys()):
        marker = "#" * level
        parts.append(f"{marker} {current_sections[level]}")
    return " > ".join(parts)


def _split_content_segments(content: str) -> list[tuple[str, str]]:
    """
    Split page content into typed segments:
    - ('table', '| ... |')
    - ('image', '> **[Figure ...]: ...')
    - ('text', 'regular text...')
    """
    segments = []
    lines = content.split('\n')
    current_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().startswith('|') and _is_table_line(lines, i):
            if current_lines:
                segments.append(("text", '\n'.join(current_lines)))
                current_lines = []
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            segments.append(("table", '\n'.join(table_lines)))
            continue

        elif line.strip().startswith('> **[Figure'):
            if current_lines:
                segments.append(("text", '\n'.join(current_lines)))
                current_lines = []
            bq_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('>'):
                bq_lines.append(lines[i])
                i += 1
            segments.append(("image", '\n'.join(bq_lines)))
            continue

        else:
            current_lines.append(line)
            i += 1

    if current_lines:
        segments.append(("text", '\n'.join(current_lines)))

    return segments


def _is_table_line(lines: list[str], idx: int) -> bool:
    """Check if current position is start of a markdown table."""
    if idx + 1 < len(lines):
        next_line = lines[idx + 1].strip()
        return bool(re.match(r'^\|[\s\-:|]+\|$', next_line))
    return False


# ── 4. Embed chunks ─────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed texts using OpenAI embeddings API in batches."""
    print("[4/5] Generating embeddings...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])
        print(f"   → Embedded batch {i//batch_size + 1} ({len(all_embeddings)}/{len(texts)})")
    return all_embeddings


# ── 5. Store in ChromaDB ────────────────────────────────────────

def store_in_chromadb(chunks: list[dict], embeddings: list[list[float]]):
    """Store chunks and embeddings in ChromaDB."""
    print("[5/5] Storing in ChromaDB...")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 166
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"   → Stored {collection.count()} chunks in ChromaDB")
    print(f"   → Persisted to: {CHROMA_PERSIST_DIR}")


# ── Main pipeline ────────────────────────────────────────────────

def ingest(pdf_path: str, output_dir: str = "parsed_output", parse_only: bool = False):
    """
    Full ingestion pipeline:
    PDF → Markdown → save locally → chunk → embed → ChromaDB
    """
    print("=" * 60)
    print("RAG PDF Ingestion Pipeline" + (" (Parse Only)" if parse_only else ""))
    print("=" * 60)

    markdown = pdf_to_markdown(pdf_path)
    md_path = save_markdown(markdown, pdf_path, output_dir)

    if parse_only:
        print("=" * 60)
        print(f"Parsing complete! Markdown saved to: {md_path}")
        print("=" * 60)
        return

    chunks = chunk_markdown(markdown)
    if not chunks:
        print("No chunks extracted. Check your PDF.")
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    store_in_chromadb(chunks, embeddings)

    print("=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest PDF into RAG pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    parser.add_argument("--output-dir", type=str, default="parsed_output",
                        help="Directory to save parsed markdown (default: parsed_output)")
    parser.add_argument("--parse-only", action="store_true",
                        help="Only convert PDF to markdown and save")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"File not found: {args.pdf}")
        exit(1)

    ingest(args.pdf, args.output_dir, args.parse_only)


if __name__ == "__main__":
    main()

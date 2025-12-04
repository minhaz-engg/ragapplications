#!/usr/bin/env python3
"""
try_lightrag.py — minimal, end‑to‑end trial of LightRAG on a markdown dataset.

What this script does
---------------------
1) Parses a markdown corpus where each product/doc looks like:

    ## Product title
    **DocID:** `daraz_123`  **Source:** Daraz
    **Category:** Laptop   **Brand:** Acer
    **Price:** ৳ 89,990   **URL:** https://...

    **Description:**
    ...long text...
yes
    ---

2) Builds a LightRAG workspace (vector index + knowledge graph).
3) Inserts each product as one document with metadata.
4) Runs example queries in different modes (naive, local, global, hybrid, mix).
5) Prints answers and (optionally) retrieved context for debugging.

Requirements
------------
- Python 3.10+
- `pip install "lightrag-hku>=1.4.9" python-dotenv`
- For OpenAI: set environment variable OPENAI_API_KEY
- For Ollama (optional): you must have Ollama running locally with an embedding model (e.g., nomic-embed-text)

Usage
-----
    python try_lightrag.py \
        --md /path/to/combined_corpus.md \
        --workdir .lightrag_demo \
        --provider openai \
        --query "Find laptops under ৳110,000 with RTX 2050 and 16GB RAM"

Tips
----
- If you change the embedding model, delete the workdir before re‑running (embedding dimensions must match).
- If you run into rate limits, lower --concurrency or set OPENAI_API_KEY to a project with higher quota.
"""
import os
import re
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# LightRAG imports
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
try:
    # Optional (only needed for the Ollama provider variant)
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc
    HAVE_OLLAMA = True
except Exception:
    HAVE_OLLAMA = False


# -----------------------------
# Markdown parsing helpers
# -----------------------------
ENTRY_SEP = re.compile(r"(?m)^---\\s*$")
FIELD_RE = {
    "doc_id": re.compile(r"\\*\\*DocID:\\*\\*\\s*`([^`]+)`"),
    "source": re.compile(r"\\*\\*Source:\\*\\*\\s*([^\\n]+)"),
    "category": re.compile(r"\\*\\*Category:\\*\\*\\s*([^\\n]+)"),
    "brand": re.compile(r"\\*\\*Brand:\\*\\*\\s*([^\\n]+)"),
    "price": re.compile(r"\\*\\*Price:\\*\\*\\s*([^\\n]+)"),
    "url": re.compile(r"\\*\\*URL:\\*\\*\\s*(\\S+)"),
}
TITLE_RE = re.compile(r"(?m)^##\\s+(.+?)\\s*$")
DESC_SPLIT_RE = re.compile(r"\\*\\*Description:\\*\\*\\s*")

def _clean(s: Optional[str]) -> Optional[str]:
    return s.strip() if isinstance(s, str) else s

def split_entries(md_text: str) -> List[str]:
    parts = [p.strip() for p in ENTRY_SEP.split(md_text) if p.strip()]
    # If file doesn't use --- separators, fall back to splitting on "## <title>"
    if len(parts) <= 1:
        # Keep the "## " line with the block
        blocks = re.split(r"(?m)^##\\s+", md_text)
        parts = [b.strip() for b in blocks if b.strip()]
    return parts

def parse_entry(block: str) -> Tuple[str, str, Dict]:
    """
    Returns: (doc_id, text_content, metadata)
    """
    # Title
    m_title = TITLE_RE.search(block)
    title = _clean(m_title.group(1)) if m_title else None

    # Fields
    fields = {}
    for k, rx in FIELD_RE.items():
        m = rx.search(block)
        fields[k] = _clean(m.group(1)) if m else None

    # Description
    desc = ""
    m_desc = DESC_SPLIT_RE.split(block, maxsplit=1)
    if len(m_desc) == 2:
        desc = m_desc[1].strip()

    # Fallback doc_id if not present
    doc_id = fields.get("doc_id") or f"md-{hash(block) & 0xFFFFFFFF:08x}"

    # Compose a single string for LightRAG insertion (rich, self‑contained)
    lines = []
    if title:
        lines.append(f"Title: {title}")
    if fields.get("brand"):
        lines.append(f"Brand: {fields['brand']}")
    if fields.get("category"):
        lines.append(f"Category: {fields['category']}")
    if fields.get("price"):
        lines.append(f"Price: {fields['price']}")
    if fields.get("source"):
        lines.append(f"Source: {fields['source']}")
    if fields.get("url"):
        lines.append(f"URL: {fields['url']}")
    if desc:
        lines.append("Description:")
        lines.append(desc)

    text_content = "\\n".join(lines)

    metadata = {
        "title": title,
        "brand": fields.get("brand"),
        "category": fields.get("category"),
        "price": fields.get("price"),
        "source": fields.get("source"),
        "url": fields.get("url"),
    }
    return doc_id, text_content, metadata

def load_markdown_corpus(path: Path) -> List[Tuple[str, str, Dict]]:
    data = path.read_text(encoding="utf-8")
    blocks = split_entries(data)
    docs = []
    for b in blocks:
        try:
            docs.append(parse_entry(b))
        except Exception as ex:
            # Keep going on parse errors, but show a short note
            print(f"[WARN] Could not parse one block (len={len(b)}): {ex}")
    return docs


# -----------------------------
# LightRAG init / insert / query
# -----------------------------
async def init_rag_openai(workdir: Path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(workdir),
        embedding_func=openai_embed,             # OpenAI embeddings
        llm_model_func=gpt_4o_mini_complete,     # OpenAI chat completion
    )
    # Required initializations in LightRAG
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def init_rag_ollama(workdir: Path, embed_model: str = "nomic-embed-text", llm_model: str = "qwen2.5:7b", num_ctx: int = 32768) -> LightRAG:
    if not HAVE_OLLAMA:
        raise RuntimeError("Ollama init requested but lightrag's Ollama extras are not installed. Install 'lightrag-hku' and ensure Ollama is running.")
    rag = LightRAG(
        working_dir=str(workdir),
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_kwargs={"options": {"num_ctx": int(num_ctx)}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  # typical dim for nomic-embed-text
            func=lambda texts: ollama_embed(texts, embed_model=embed_model),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def insert_docs(rag: LightRAG, docs: List[Tuple[str, str, Dict]], concurrency: int = 3) -> None:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _worker(doc_id: str, text: str, meta: Dict):
        async with sem:
            try:
                await rag.ainsert(text, doc_id=doc_id, metadata=meta)
                print(f"[OK] inserted: {doc_id}")
            except Exception as e:
                print(f"[ERR] insert failed for {doc_id}: {e}")

    tasks = [_worker(d, t, m) for (d, t, m) in docs]
    for chunk in [tasks[i:i+50] for i in range(0, len(tasks), 50)]:
        await asyncio.gather(*chunk)

async def run_queries(rag: LightRAG, queries: List[str], mode: str = "mix", show_context: bool = False, include_refs: bool = True):
    for q in queries:
        print("\\n" + "="*90)
        print(f"Q: {q}")
        if show_context:
            ctx = await rag.aquery(q, param=QueryParam(mode=mode, only_need_context=True))
            print("\\n[Retrieved context]\\n")
            print(ctx)

        ans = await rag.aquery(q, param=QueryParam(mode=mode, include_references=include_refs))
        print("\\n[Answer]\\n")
        print(ans)
        print("="*90 + "\\n")


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Try LightRAG on a markdown corpus.")
    p.add_argument("--md", type=Path, required=True, help="Path to markdown corpus (e.g., combined_corpus.md)")
    p.add_argument("--workdir", type=Path, default=Path(".lightrag_demo"), help="Directory to store LightRAG data")
    p.add_argument("--provider", choices=["openai", "ollama"], default="openai", help="Model provider to use")
    p.add_argument("--ollama-llm", default="qwen2.5:7b", help="Ollama LLM model name")
    p.add_argument("--ollama-embed", default="nomic-embed-text", help="Ollama embedding model name")
    p.add_argument("--num-ctx", type=int, default=32768, help="Context window to request from Ollama models")
    p.add_argument("--concurrency", type=int, default=3, help="Max concurrent insert tasks")
    p.add_argument("--mode", default="mix", help="Query mode: naive | local | global | hybrid | mix | bypass")
    p.add_argument("--query", action="append", help="Query to run (can pass multiple). If omitted, a few examples are used.")
    p.add_argument("--show-context", action="store_true", help="Print retrieved context (no LLM generation) before answers")
    p.add_argument("--reset", action="store_true", help="Delete workdir before running (e.g., when changing embedding models)")
    return p

async def amain(args):
    if args.reset and args.workdir.exists():
        for p in args.workdir.glob("*"):
            try:
                p.unlink()
                print(f"[clean] removed: {p}")
            except Exception:
                pass

    args.workdir.mkdir(parents=True, exist_ok=True)

    # Init provider
    if args.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("Please export OPENAI_API_KEY before running with provider=openai")
        rag = await init_rag_openai(args.workdir)
    else:
        rag = await init_rag_ollama(args.workdir, embed_model=args.ollama_embed, llm_model=args.ollama_llm, num_ctx=args.num_ctx)

    # Load docs
    docs = load_markdown_corpus(args.md)
    print(f"Parsed {len(docs)} entries from {args.md}")

    # Insert
    await insert_docs(rag, docs, concurrency=args.concurrency)

    # Queries
    queries = args.query or [
        "List the best mid-range gaming laptops under ৳110,000 with RTX 2050 and 16GB RAM.",
        "Show budget choices from StarTech with at least 512GB SSD and 15.6-inch display.",
        "Compare any two Acer Swift models in the dataset; what are the main differences?",
    ]
    await run_queries(rag, queries, mode=args.mode, show_context=args.show_context, include_refs=True)

def main():
    parser = build_argparser()
    args = parser.parse_args()
    asyncio.run(amain(args))

if __name__ == "__main__":
    main()
